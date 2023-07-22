import torch
import torch.nn.functional as F
from net_utils.initialize import *
from net_utils.variational import reparameterize

# C. P. Burgess, I. Higgins, A. Pal, L. Matthey, N. Watters, G. Desjardins, and A. Lerchner.
# Understanding disentangling in β-vae. arXiv preprint arXiv:1804.03599, 2018.
#
# I. Higgins, L. Matthey, A. Pal, C. Burgess, X. Glorot, M. Botvinick, S. Mohamed, and A. Ler- chner. beta-vae:
# Learning basic visual concepts with a constrained variational framework.
# In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017,


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAE_H(nn.Module):
    """
    code from  https://github.com/1Konny/Beta-VAE/
    License:
        MIT License
    Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017).
    ''' changed z_dim to 32 as in celeb_A setup, and nc to 1 for medical images'''
    """
    def __init__(self, z_dim=32, nc=1, additional_layer=True):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        if additional_layer:
            self.encoder = nn.Sequential(
                nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
                nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
                nn.ReLU(True),
                ####  Added for larger inputs
                nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
                nn.ReLU(True),
                ######
                nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
                nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
                nn.ReLU(True),
                nn.Conv2d(32, 256, 4, 1),            # B, 256,  1,  1
                nn.ReLU(True),
                View((-1, 256*1*1)),                 # B, 256
                nn.Linear(256, z_dim*2),             # B, z_dim*2
            )
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, 256),  # B, 256
                View((-1, 256, 1, 1)),  # B, 256,  1,  1
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 32, 4),  # B,  32,  4,  4
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
                nn.ReLU(True),
                ####  Added for larger inputs
                nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
                nn.ReLU(True),
                ####
                nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
                nn.ReLU(True),
                nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
                nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
                nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
                nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
                nn.ReLU(True),
                nn.Conv2d(32, 256, 4, 1),            # B, 256,  1,  1
                nn.ReLU(True),
                View((-1, 256*1*1)),                 # B, 256
                nn.Linear(256, z_dim*2),             # B, z_dim*2
            )
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, 256),  # B, 256
                View((-1, 256, 1, 1)),  # B, 256,  1,  1
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 32, 4),  # B,  32,  4,  4
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32,  8,  8
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
                nn.ReLU(True),
                nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
            )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparameterize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, {'z_mu': mu, 'z_logvar': logvar}

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class VAEHigLoss:
    def __init__(self, beta=4, gamma=10.0, max_capacity=25, loss_type='B'):
        super(VAEHigLoss, self).__init__()
        self.beta = beta
        self.num_iter = 0
        self.max_capacity = max_capacity
        self.Capacity_max_iter = 1e5
        self.C_max = torch.Tensor([self.max_capacity])
        self.C_stop_iter = self.Capacity_max_iter
        self.loss_type = loss_type
        self.gamma=gamma

    def __call__(self, x_recon, x, z):
        self.num_iter += 1
        mu = z['z_mu']
        log_var = z['z_logvar']
        kld_weight = 0.008 # 64/8000  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(x_recon, x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(x.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return loss

if __name__ == '__main__':
    pass