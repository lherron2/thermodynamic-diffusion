from torch.utils.data import Dataset
from torch import vmap
import torch.nn.functional as F
import torch.nn as nn
import torch
from utils import *


def polynomial_noise(t, alpha_max, alpha_min, s=1e-5):
    """
    Same schedule used in Hoogeboom et. al. (Equivariant Diffusion for Molecule Generation in 3D)
    """
    T = t[-1]
    alphas = (1-2*s)*(1-(t/T)**2) + s
    a = alphas[1:]/alphas[:-1]
    a[a**2 < 0.001] = 0.001
    alpha_schedule = torch.cumprod(a, 0)
    return alpha_schedule

NOISE_FUNCS = {
    "polynomial": polynomial_noise,
              }
class DiffusionProcess:
    """
    Instantiates the noise parameterization, rescaling of noise distribution, and
    timesteps for a diffusion process.
    """
    def __init__(self,
                 num_diffusion_timesteps,

                 noise_schedule,
                 alpha_max,
                 alpha_min,
                 NOISE_FUNCS,
                ):

        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.times = torch.arange(num_diffusion_timesteps)
        self.alphas = NOISE_FUNCS[noise_schedule](torch.arange(num_diffusion_timesteps+1),
                                                  alpha_max,
                                                  alpha_min)

class VPDiffusion(DiffusionProcess):
    """
    Subclass of a DiffusionProcess: Performs a diffusion according to the VP-SDE.
    """
    def __init__(self,
                 num_diffusion_timesteps,
                 noise_schedule="polynomial",
                 alpha_max=20.,
                 alpha_min=0.01,
                 NOISE_FUNCS=NOISE_FUNCS,
                ):

        super().__init__(num_diffusion_timesteps,
                         noise_schedule,
                         alpha_max,
                         alpha_min,
                         NOISE_FUNCS
                        )

        self.bmul = vmap(torch.mul)

    def get_alphas(self):
        return self.alphas

    def forward_kernel(self, x0, t, scale):
        """
        Maginal transtion kernels of the forward process. q(x_t|x_0).
        """
        alphas_t = self.alphas[t]
        noise = self.bmul(torch.randn_like(x0), scale)
        x_t = self.bmul(x0, alphas_t.sqrt()) + self.bmul(noise, (1-alphas_t).sqrt())
        return x_t, noise

    def reverse_kernel(self, x_t, t, backbone, pred_type, data_in=None):
        """
        Marginal transition kernels of the reverse process. p(x_0|x_t).
        """
        data_in = default(x_t, data_in)
        alphas_t = self.alphas[t]
        if pred_type == "noise":
            noise = backbone(data_in, alphas_t)
            noise_interp = self.bmul(noise, (1-alphas_t).sqrt())
            x0_t = self.bmul((x_t - noise_interp), 1/alphas_t.sqrt())
        elif pred_type == "x0":
            x0_t = backbone(data_in, alphas_t)
            x0_interp = self.bmul(x0_t, (alphas_t).sqrt())
            noise = self.bmul((x_t - x0_interp), 1/(1-alphas_t).sqrt())
        else:
            raise Exception("Please provide a valid prediction type: 'noise' or 'x0'")

        return x0_t, noise

    def reverse_step(self, x_t, t, t_next, backbone, pred_type, data_in=None):
        """
        Stepwise transition kernel of the reverse process p(x_t-1|x_t).
        """

        alphas_t = self.alphas[t]
        alphas_t_next = self.alphas[t_next]
        data_in = default(x_t, data_in)
        x0_t, noise = self.reverse_kernel(x_t, t, backbone, pred_type, data_in=data_in)
        xt_next = self.bmul(alphas_t_next.sqrt(), x0_t) + self.bmul((1-alphas_t_next).sqrt(), noise)
        return xt_next

    def sample_prior(self, xt, scale):
        """
        Generates a sample from a prior distribution p(z) ~ p(x_T).
        """
        noise = self.bmul(torch.randn_like(xt), scale)
        return noise

    def compute_SNR(self, t):
        alpha_sq = self.alphas[t.long()].pow(2)
        sigma_sq = 1 - alpha_sq
        gamma_t = -(torch.log(alpha_sq) - torch.log(sigma_sq))
        return torch.exp(-gamma_t)
