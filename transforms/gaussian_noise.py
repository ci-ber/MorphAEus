import torch
import numpy as np


class GaussianNoise:
    # from https://github.com/AntanasKascenas/DenoisingAE/
    def __init__(self, noise_std=0.2, noise_res=16):
        super(GaussianNoise, self).__init__()
        self.noise_std = noise_std
        self.noise_res = noise_res

    def __call__(self, x):

        ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], self.noise_res, self.noise_res),
                          std=self.noise_std).to(x.device)

        ns = torch.nn.functional.upsample_bilinear(ns, size=[x.shape[2], x.shape[3]])

        # Roll to randomly translate the generated noise.
        roll_x = np.random.choice(range(x.shape[2]))
        roll_y = np.random.choice(range(x.shape[3]))
        ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])

        mask = x.sum(dim=1, keepdim=True) > 0.01
        ns *= mask  # Only apply the noise in the foreground.
        res = x + ns
        return res
