from torch.utils.data import Dataset
from torch import vmap
import torch.nn.functional as F
import torch.nn as nn
import torch
from utils import *

def temperature_density_rescaling(std_temp, ref_temp):
    return (std_temp/ref_temp).pow(1.5)

def identity(t, *args, **kwargs):
    return t

RESCALE_FUNCS = {
    "density": temperature_density_rescaling,
    "no_rescale": identity,
                }

class DiffusionModel:
    """
    A DiffusionModel consists of instances of a DiffusionProcess, Backbone,
    Loader, and Directory objects.
    """
    def __init__(self,
                 diffusion_process,
                 backbone,
                 loader,
                 directory,
                 pred_type,
                 control_ref=310,
                 rescale_func_name="no_rescale",
                 RESCALE_FUNCS=RESCALE_FUNCS,
                ):

        self.loader = loader
        self.BB = backbone
        self.DP = diffusion_process
        self.directory = directory
        self.control_ref = control_ref
        self.pred_type = pred_type
        self.rescale_func = RESCALE_FUNCS[rescale_func_name]

    def noise_batch(self, b_t, t, control):
        """
        Wrapper which calls applies the (marginal) transition kernel
        of the forward noising process.

        Wrapper to allow the alphas to be sampled and reshaped.
        """
        scale = self.rescale_func(control, self.control_ref)
        return self.DP.forward_kernel(b_t, t, scale)

    def denoise_batch(self, b_t, t, data_in=None):
        """
        Wrapper which calls applies the (marginal) transition kernel
        of the reverse noising process.

        Wrapper to allow the alphas to be sampled and reshaped.
        """
        data_in = default(b_t, data_in)
        return self.DP.reverse_kernel(b_t, t, self.BB,
                                      self.pred_type, data_in=data_in)

    def denoise_step(self, b_t, t, t_next, control=None, data_in=None):
        """
        Wrapper which calls applies the (marginal) transition kernel
        of the reverse noising process.

        Wrapper to allow the alphas to be sampled and reshaped.
        """

        data_in = default(b_t, data_in)
        b_t_next = self.DP.reverse_step(b_t, t, t_next, self.BB,
                                        self.pred_type, data_in=data_in)
        if control is not None:
            scale = self.rescale_func(control, self.control_ref)
            b_t_next[self.loader.control_slice] = control

        return b_t_next

    def sample_prior(self, batch_size, dims, unstd_control=None, std_control=None):
        if unstd_control is not None:
            scale = self.rescale_func(unstd_control, self.control_ref)
            prior_sample =  torch.randn(batch_size, *dims, dtype=torch.float)*scale
            prior_sample[self.loader.control_slice] = std_control
        else:
            prior_sample =  torch.randn(batch_size, *dims, dtype=torch.float)*scale

        return prior_sample

    def sample_times(self, num_times):
        """
        Randomly sample times from the time-discretization of the
        diffusion process
        """
        return torch.randint(low=0,
                  high=self.DP.num_diffusion_timesteps,
                  size=(num_times,)).long()

    @staticmethod
    def get_adjacent_times(times):
        """
        Pairs t with t+1 for all times in the time-discretization
        of the diffusion process.
        """
        times_next = torch.cat((torch.Tensor([0]).long(), times[:-1]))
        return list(zip(reversed(times), reversed(times_next)))

class DiffusionTrainer(DiffusionModel):
    """
    Subclass of a DiffusionModel: A trainer defines a loss function and
    performs backprop + optimizes model outputs.
    """
    # how to make DiffusionTrainer take all the same arguments as DiffusionModel
    def __init__(self,
                 diffusion_process,
                 backbone,
                 loader,
                 directory,
                 pred_type,
                 optim=None,
                 scheduler=None,
                 control_ref=310,
                 rescale_func_name="density",
                 RESCALE_FUNCS=RESCALE_FUNCS,
                ):

        super().__init__(diffusion_process,
                         backbone,
                         loader,
                         directory,
                         pred_type,
                         control_ref,
                         rescale_func_name,
                         RESCALE_FUNCS)

    def loss_function(self, e, e_pred, weight, loss_type="l2"):
        """
        loss function can be the l1-norm, l2-norm, or the VLB (weighted l2-norm)
        """

        sum_indices = tuple(list(range(1,self.loader.num_dims)))

        def l1_loss(e, e_pred, weight):
            return (e - e_pred).abs().sum(sum_indices)

        def l2_loss(e, e_pred, weight):
            return (e - e_pred).pow(2).sum((1,2,3)).pow(0.5).mean()

        def VLB_loss(e, e_pred, weight):
            return (weight*((e - e_pred).pow(2).sum(sum_indices)).pow(0.5)).mean()

        loss_dict = {
            "l1": l1_loss,
            "l2": l2_loss,
            "VLB": VLB_loss
                    }

        return loss_dict[loss_type](e, e_pred, weight)

    def train(self, num_epochs, grad_accumulation_steps=1, print_freq=10, batch_size=128, loss_type="l2"):
        """
        Trains a diffusion model.
        """

        train_loader = torch.utils.data.DataLoader(
            self.loader,
            batch_size=batch_size,
            shuffle=True,
        )

        for epoch in range(num_epochs):
            for i, (unstd_control, std_control, b) in enumerate(train_loader, 0):
                t = self.sample_times(b.size(0))
                t_prev = t - 1
                t_prev[t_prev == -1] = 0
                weight = self.DP.compute_SNR(t_prev) - self.DP.compute_SNR(t)
                b_t, noise = self.noise_batch(b, t, unstd_control)
                b_0, noise_pred = self.denoise_batch(b_t, t)
                loss = self.loss_function(noise, noise_pred, weight, loss_type=loss_type)/grad_accumulation_steps

                if i % grad_accumulation_steps == 0:
                    self.BB.optim.zero_grad()
                    loss.backward()
                    self.BB.optim.step()
                    self.BB.scheduler.step()

                if i % print_freq == 0:
                    print(f"step: {i}, loss {loss.detach():.3f}")
            print(f"epoch: {epoch}")
            if self.BB.scheduler:
                self.BB.scheduler.step()

            self.BB.save_state(self.directory, epoch)

class DiffusionSampler(DiffusionModel):
    """
    Subclass of a DiffusionModel: A sampler generates samples from random noise.
    """
    def __init__(self,
                 diffusion_process,
                 backbone,
                 loader,
                 directory,
                 pred_type,
                 control_ref=310,
                 rescale_func_name="density",
                 RESCALE_FUNCS=RESCALE_FUNCS,
                ):

        super().__init__(diffusion_process,
                         backbone,
                         loader,
                         directory,
                         pred_type,
                         control_ref,
                         rescale_func_name,
                         RESCALE_FUNCS)

    def sample_batch(self, batch_size, unstd_control):

        std_control = self.loader.standardize(unstd_control)

        xt = self.sample_prior(batch_size,
                               self.loader.get_all_but_batch_dim(),
                               unstd_control=unstd_control, # for rescaling noise
                               std_control=std_control, # for guiding tensor
                               )

        time_pairs = self.get_adjacent_times(self.DP.times)

        for t, t_next in time_pairs:
            print(t, t_next)
            t = torch.Tensor.repeat(t, batch_size)
            t_next = torch.Tensor.repeat(t_next, batch_size)

            xt_next = self.denoise_step(xt, t, t_next, std_control)
            xt = xt_next
        return xt

    def save_batch(self, batch, save_prefix, temperature, save_idx):
        save_path = os.path.join(self.directory.sample_path, f"{int(temperature)}K")
        os.makedirs(save_path, exist_ok=True)
        np.savez_compressed(os.path.join(save_path, f"{save_prefix}_idx={save_idx}.npz"), traj=batch)

    def sample_loop(self, num_samples, batch_size, save_prefix, temperature):
        n_runs = max(num_samples//batch_size, 1)
        if num_samples//batch_size == 0:
            batch_size = num_samples
        with torch.no_grad():
            for save_idx in range(n_runs):
                unstd_control = torch.Tensor([temperature])
                x0 = self.sample_batch(batch_size, unstd_control)
                self.save_batch(x0[:,:-1,:,:], save_prefix, temperature, save_idx)

