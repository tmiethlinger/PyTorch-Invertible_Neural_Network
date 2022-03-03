import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

import torch
import torch.nn as nn
import torch.optim
from torch.autograd import grad

from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules import GLOWCouplingBlock, PermuteRandom

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Based on "Analyzing Inverse Problems with Invertible Neural Networks" by L. Ardizzone et al.
class InvertibleNeuralNetwork():
    def __init__(self, ndim_tot, ndim_y, ndim_x, ndim_z, num_blocks, feature, sub_len, batch_size, l2_reg, y_noise_scale, zeros_noise_scale, lambd_predict, lambd_predict_back, lambd_latent, lambd_rev, loss_back='mse', seed=1):
        self.sub_len = sub_len
        self.loss_back = loss_back
        self.ndim_tot = ndim_tot
        self.ndim_y = ndim_y
        self.ndim_x = ndim_x
        self.ndim_z = ndim_z
        self.feature = feature
        self.num_blocks = num_blocks
        torch.manual_seed(seed)
        
        # Set up subnetwork
        def subnet_fc(c_in, c_out):
            layers = []
            layers.append(nn.Linear(c_in, self.feature))
            layers.append(nn.Tanh())
            for i in range(self.sub_len):
                layers.append(nn.Linear(self.feature, self.feature))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(self.feature, c_out))
            return nn.Sequential(*layers)
        
        nodes = [InputNode(self.ndim_tot, name='input')]

        for k in range(self.num_blocks):
            nodes.append(Node(nodes[-1],
                              GLOWCouplingBlock,
                              {'subnet_constructor':subnet_fc, 'clamp':2.0},
                              name=F'coupling_{k}'))
            nodes.append(Node(nodes[-1], 
                              PermuteRandom,
                              {'seed':k},
                              name=F'permute_{k}'))

        nodes.append(OutputNode(nodes[-1], name='output'))

        self.model = ReversibleGraphNet(nodes, verbose=False)
        
        # Training parameters
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.y_noise_scale = y_noise_scale
        self.zeros_noise_scale = zeros_noise_scale

        
        # relative weighting of losses:
        self.lambd_predict = lambd_predict
        self.lambd_predict_back = lambd_predict_back
        self.lambd_latent = lambd_latent
        self.lambd_rev = lambd_rev
        
        # Zero padding dimensions if both sides don't match [x,0] <-> [z,y,0]
        self.pad_x = torch.zeros(self.batch_size, ndim_tot - ndim_x)
        self.pad_yz = torch.zeros(self.batch_size, ndim_tot - ndim_y - ndim_z)
        
        # Set up the optimizer
        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        
        # MMD losses
        self.loss_backward = self.MMD_multiscale
        self.loss_latent = self.MMD_multiscale
        
        # Supervised loss for the y,z -> x direction
        self.loss_fit = torch.nn.MSELoss()
        
        # Initialize weights
        for param in self.trainable_parameters:
            param.data = 0.05*torch.randn_like(param)            
        self.model.to(device);
        return

    # A function which to make the supervised loss x <- y,z choosable
    def loss_fit_back(self, output, gt):
        if self.loss_back == "l1":
            return torch.nn.L1Loss()(output, gt)
        if self.loss_back == "l1_ssim":
            return torch.nn.L1Loss()(output, gt) + self.ssim(output.unsqueeze(0).unsqueeze(0), gt.unsqueeze(0).unsqueeze(0))
        if self.loss_back == "mse_ssim":
            return torch.nn.MSELoss()(output, gt) + self.ssim(output.unsqueeze(0).unsqueeze(0), gt.unsqueeze(0).unsqueeze(0))
        else:
            return torch.nn.MSELoss()(output, gt)
        
    def MMD_multiscale(self, x, y, scale = 1):
        xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2.*xx
        dyy = ry.t() + ry - 2.*yy
        dxy = rx.t() + ry - 2.*zz

        XX, YY, XY = (torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device),
                      torch.zeros(xx.shape).to(device))

        for a in [0.05*scale, 0.2*scale, 0.9*scale]:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

        return torch.mean(XX + YY - 2.*XY)

    def train(self, lr, n_epochs, train_loader, val_loader, log_writer=None):
        self.n_epochs = n_epochs
        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=lr, eps=1e-6, weight_decay=self.l2_reg)
        
        print("Epoch mse_y_test l_x_1 l_x_2 l_y l_z")

        tensor_x_val_gt = torch.cat([x for x,y in val_loader], dim=0)
        tensor_y_val_gt = torch.cat([y for x,y in val_loader], dim=0)

        for i_epoch in range(n_epochs):
            loss, l_y, l_z, l_x_1, l_x_2 = self.train_epoch(train_loader, i_epoch)
            
            self.model.eval()
            if i_epoch % 100 == 0:

                self.pad_x = self.zeros_noise_scale * torch.randn(self.batch_size, self.ndim_tot - self.ndim_x, device=device)
                tensor_x_pad_val_gt = torch.cat((tensor_x_val_gt, self.pad_x),  dim=1)
                tensor_y_val_pred = self.model(tensor_x_pad_val_gt)[0]
                mseloss_y = torch.nn.MSELoss()(tensor_y_val_gt, tensor_y_val_pred[:, -self.ndim_y:])
                print("{0} {1:0.4f} {2:0.4f} {3:0.4f} {4:0.4f} {5:0.4f}".format(i_epoch, mseloss_y, l_x_1, l_x_2, l_y, l_z))
        return
    
    def train_epoch(self, train_loader, i_epoch=0):
        self.model.train()

        l_tot = 0
        l_y = 0
        l_z = 0
        l_x_1 = 0
        l_x_2 = 0
        batch_idx = 1

        # If MMD on x-space is present from the start, the self.model can get stuck.
        # Instead, ramp it up exponentially.  
        # loss_factor = 1
        loss_factor = min(1., 2. * 0.002**(1. - (float(i_epoch) / self.n_epochs)))
        for x, y in train_loader:
                        
            #Turn
            y_clean = y.clone()
            self.pad_x = self.zeros_noise_scale * torch.randn(self.batch_size, self.ndim_tot - self.ndim_x, device=device)
            self.pad_yz = self.zeros_noise_scale * torch.randn(self.batch_size, self.ndim_tot - self.ndim_y - self.ndim_z, device=device)
            self.optimizer.zero_grad()
            y += self.y_noise_scale * torch.randn(self.batch_size, self.ndim_y, dtype=torch.float, device=device)
            x = torch.cat((x, self.pad_x),  dim=1)
            y = torch.cat((torch.randn(self.batch_size, self.ndim_z, device=device), self.pad_yz, y), dim=1)
            
            # Forward step:
            output = self.model(x)[0]

            # Shorten output, and remove gradients wrt y, for latent loss
            y_short = torch.cat((y[:, :self.ndim_z], y[:, -self.ndim_y:]), dim=1)
            l = self.lambd_predict * self.loss_fit(output[:, self.ndim_z:], y[:, self.ndim_z:])
            l_y += l.data.item()
            
            output_block_grad = torch.cat((output[:, :self.ndim_z], output[:, -self.ndim_y:].data), dim=1)

            l_latent = self.lambd_latent * self.loss_latent(output_block_grad, y_short)
            l += l_latent
            l_tot += l.data.item()
            l_z += l_latent.data.item()
            l.backward(retain_graph=True)
            
            # Backward step:
            self.pad_yz = self.zeros_noise_scale * torch.randn(self.batch_size, self.ndim_tot - self.ndim_y - self.ndim_z, device=device)
            y = y_clean
            orig_z_perturbed = output.data[:, :self.ndim_z] + self.y_noise_scale * torch.randn(self.batch_size, self.ndim_z, device=device)
            y_rev = torch.cat((orig_z_perturbed, self.pad_yz, y), dim=1)
            y_rev_rand = torch.cat((torch.randn(self.batch_size, self.ndim_z, device=device), self.pad_yz, y), dim=1)
            
            output_rev = self.model(y_rev, rev=True)[0]
            output_rev_rand = self.model(y_rev_rand, rev=True)[0]
            
            l_rev = self.lambd_rev * loss_factor * self.loss_backward(output_rev_rand[:, :self.ndim_x], x[:, :self.ndim_x])
            
            l_x_1 += l_rev.data.item()

            l_rev_2 = self.lambd_predict_back * self.loss_fit_back(output_rev, x)
            l_rev += l_rev_2 
            l_x_2 += l_rev_2.data.item()
                       
            l_tot += l_rev.data.item()

            l_rev.backward()
            self.optimizer.step()

            batch_idx += 1

        return l_tot/batch_idx, l_y/batch_idx, l_z/batch_idx, l_x_1/batch_idx, l_x_2/batch_idx


    def save_model(self, path):

        torch.save(self.model.state_dict(), path + ".pt")


    def load_model(self, path):        
        
        self.model.load_state_dict(torch.load(path + ".pt"))
        self.model.eval()
        return self
