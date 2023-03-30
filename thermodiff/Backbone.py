from torch.utils.data import Dataset
from torch import vmap
import torch.nn.functional as F
import torch.nn as nn
import torch
from utils import *
from torch.optim.lr_scheduler import MultiStepLR

class Backbone(nn.Module):
    """
    Diffusion wrapper for instances of deep learning architectures.
    """
    def __init__(self,
                 model,
                 data_shape,
                 target_shape,
                 num_dims=3,
                 lr=1e-3,
                 optim=None,
                 scheduler=None,
                ):

        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(self.device)
        data_shape = tuple([data_shape] * (num_dims-2)) # ignore batch and channel dims
        target_shape = tuple([target_shape] * (num_dims-2))

        self.interp = Interpolater(data_shape, target_shape)
        dim_vec = torch.ones(num_dims)
        dim_vec[0] = -1
        self.expand_batch_to_dims = tuple(dim_vec)
        self.state = None
        self.start_epoch = 0

        optim_dict = {"Adam": torch.optim.Adam(self.model.parameters(),
                                               lr=lr,
                                               weight_decay=False,
                                               betas=(0.9, 0.99),
                                               amsgrad=True, eps=1e-9)
                     }



        self.optim = default(optim_dict["Adam"], optim)

        scheduler_dict = {"multistep": MultiStepLR(self.optim,
                                                   milestones=[5],
                                                   gamma=0.1),
                         }

        self.scheduler = default(scheduler_dict["multistep"], scheduler)

    @staticmethod
    def get_model_path(directory, epoch):
        return os.path.join(directory.model_path, f"{directory.identifier}_{epoch}.pt")

    def save_state(self, directory, epoch):
        """
        saves internal state of the backbone model.
        """
        states = {"model": self.model.state_dict(), "optim": self.optim.state_dict(), "epoch": epoch}
        os.makedirs(directory.model_path, exist_ok=True)
        save_path = self.get_model_path(directory, epoch)
        torch.save(states, save_path)

    def load_state(self, directory, epoch):
        """
        loads internal state of the backbone model.
        """
        state_dict = torch.load(self.get_model_path(directory, epoch),
                                map_location=torch.device(directory.device))
        return state_dict

    def load_model(self, directory, epoch):
        """
        Loads model, optimizer, and starting epoch from state dict.
        """
        state_dict = self.load_state(directory, epoch)
        self.model.load_state_dict(state_dict["model"])
        self.optim.load_state_dict(state_dict["optim"])
        self.start_epoch = int(state_dict['epoch'])+1

class ConvBackbone(Backbone):
    """
    Backbone with a forward method for Convolutional Networks
    """
    def __init__(self,
                 model,
                 data_shape,
                 target_shape,
                 num_dims=4,
                 lr=1e-3,
                 optim=None,
                 eval_mode='train',
                 self_condition=True,
                 scheduler=None):

        super().__init__(model,
                         data_shape,
                         target_shape,
                         num_dims,
                         lr,
                         optim,
                         scheduler)

        self.eval_mode = eval_mode
        self.self_condition = True

    def get_self_condition(self, data, t):
        if self.eval_mode == 'train' and self.self_condition == True:
            if torch.rand(1) < 0.5:
                with torch.no_grad():
                    return self.model(data.to(self.device), t.to(self.device))
            else:
                return None
        elif self.eval_mode == 'sample' and self.self_condition == True:
            return self.model(data.to(self.device), t.to(self.device))
        else:
            return None

    def forward(self, batch, t):
        upsampled = self.interp.to_target(batch)

        self_condition = self.get_self_condition(upsampled, t)
        upsampled_out = self.model(upsampled.to(self.device),
                                   t.to(self.device),
                                   x_self_cond=self_condition)
        batch_out = self.interp.from_target(upsampled_out.to("cpu"))
        return batch_out

class GraphBackbone(Backbone):
    """
    Backbone with a forward method for Convolutional Networks
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, batch, t):

        # Implement DGL compatible forward pass
        upsampled = self.interp.to_target(batch)
        upsampled_out = self.model(batch, t)
        batch_out = self.inter.from_target(upsampled_out)
        return batch_out
