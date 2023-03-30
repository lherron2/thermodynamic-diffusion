from torch.utils.data import Dataset
from torch import vmap
import torch.nn.functional as F
import torch.nn as nn
import torch
from utils import *
import numpy as np

class Transform:
    def __init__(self, data, dim, pos):
        pass

class WhitenTransform(Transform):
    def __init__(self, data, dim, pos):
        # Selects out a dimension and position within dimension to standardize.
        # For example, if the control parameter is the last position in the channel
        # dimension then dim = 1, pos = -1.
        super().__init__(data, dim, pos)
        self.mean = data.mean(0)[pos]
        self.std = data.std(0)[pos]

    def forward(self, x):
        return (x - self.mean)/self.std

    def reverse(self, x):
        return x * self.std + self.mean

class MinMaxTransform(Transform):
    def __init__(self, data, dim, pos):
        super().__init__(data, dim, pos)

        self.min_data = data.min(0)[pos]
        self.max_data = data.max(0)[pos]

    def forward(self, x):
        return (x - self.min_data)/2*(self.max_data - self.min_data)

    def reverse(self, x):
        return 2*(self.max_data - self.min_data)*x + self.min_data

class IdentityTransform(Transform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x

    def reverse(self, x):
        return x

TRANSFORMS = {
    "whiten": WhitenTransform,
    "min_max": MinMaxTransform,
    "identity": IdentityTransform
}

class Dequantizer:
    def __init__(self, scale):
        self.scale = scale

class NormalDequantization(Dequantizer):
    def __init__(self, scale):
        super().__init__(scale)

    def forward(self, x):
        return x+ torch.randn(*x.shape)*self.scale

class UniformDequantization(Dequantizer):
    def __init__(self, scale):
        super().__init__(scale)

    def forward(self, x):
        return x+ torch.rand(*x.shape)*self.scale

DEQUANTIZERS = {
    "normal": NormalDequantization,
    "uniform": UniformDequantization
}

class Loader(Dataset):
    def __init__(self,
                 directory: Directory,
                 num_dims: int = 4,
                 transform_type: str = "whiten",
                 control_tuple: tuple = (1, -1),
                 dequantize: bool = True,
                 dequantize_type: str = "normal",
                 dequantize_scale: float = 1e-2,
                 TRANSFORMS: dict = TRANSFORMS,
                 DEQUANTIZERS: dict = DEQUANTIZERS,
                ):


        # load data from npz file using path from Directory.
        self.directory = directory
        self.data = torch.from_numpy(np.load(self.directory.get_dataset_path())).float()
        if dequantize:
            self.dequantizer = DEQUANTIZERS[dequantize_type](dequantize_scale)
            self.data = self.dequantize(self.data)

        self.data_dim = self.data.shape[-1]
        self.num_channels = self.data.shape[1]
        self.num_dims = len(self.data.shape)


        # building slice object to retrieve control params from Tensor
        (self.control_slice,
         self.control_dim,
         self.control_pos) = self.build_control_slice(control_tuple,
                                                      num_dims)
        self.batch_slice = self.build_batch_slice(num_dims)

        self.transform = TRANSFORMS[transform_type](self.data,
                                                    self.control_dim,
                                                    self.control_pos)

        self.unstd_control = self.data[self.control_slice][self.batch_slice]
        self.std_control = self.standardize(self.data[self.control_slice])[self.batch_slice]

    def build_control_slice(self, control_tuple, data_dim):
        """
        Builds a slice object which retrieves the control parameters from the tensor. For example,
        for a 4D Tensor (b, c, x, y) whose diffusion is controlled via the value of the data in the
        last position of the channel dimension, the slice object will be [:,-1,:,:].

        The control tuple contains (dim, pos) where dim is the dimension of the tensor which contains
        the control parameters and pos is the position of the control parameter along dim. The default
        value of (1,-1) assumes that for a tensor shaped as (b, c, x, y) the last position along the
        channel dimension corresponds to the control variable.

        For now only one slice of the tensor is able to be controlled, but this functionality may
        be extended to controlling muliple slices.
        """
        (control_dim, control_pos) = control_tuple

        if control_dim == None:
            control_slice = [slice(None) for dim in range(data_dim)]
        else:
            control_slice = [slice(None,None) for dim in range(data_dim)]
            # if the control slice is at the last position along the control dimension
            # then we have to build the slice object differently (i.e. [:,...,-1,...,:])
            # requires a different slice form than [:,...,0,...,:].
            if control_pos == -1:
                control_slice[control_dim] = slice(control_pos,None)
            else:
                control_slice[control_dim] = slice(control_pos,control_pos+1)

        return tuple(control_slice), control_dim, control_pos

    def build_batch_slice(self, data_dim, batch_dim=0):
        """
        Preserves the batch dimension of a tensor while taking the first element along
        the other dimensions. This is useful for cases where the control parameter is
        identical along all of the remaining dimensions (saves memory).
        """
        batch_slice = [slice(0,1) for dim in range(data_dim)]
        batch_slice[batch_dim] = slice(None,None)
        return batch_slice

    def dequantize(self, x):
        """
        Calls the dequantization method defined in DEQUANTIZERS
        """
        return self.dequantizer.forward(x)

    def standardize(self, x):
        """
        Calls the standardizing transform defined in TRANSFORMS
        """
        return self.transform.forward(x)

    def unstandardize(self, x):
        """
        Calls the inverse of the standardizing transform defined in TRANSFORMS
        """
        return self.transform.reverse(x)

    def get_data_dim(self):
        return self.data_dim

    def get_num_dims(self):
        return self.num_dims

    def get_num_channels(self):
        return self.num_channels

    def get_all_but_batch_dim(self):
        return self.data.shape[1:]

    def get_batch(self, index):
        """
        Used for testing purposes.
        """
        x = self.data[index:index+1]
        std_control = self.std_control[index:index+1]
        x[self.control_slice] = std_control
        return x.float()

    def __getitem__(self, index):
        """
        Returns a tuple consisting of:
        (standardized_control_params at index (Tensor),
        unstandardized_control_params at index (Tensor),
        data at index (Tensor))

        The returned tensors must have dimension data_dim - 1.
        """
        x = torch.clone(self.data[index:index+1])
        unstd_control = self.unstd_control[index:index+1]
        std_control = self.std_control[index:index+1]
        x[self.control_slice] = std_control
        return unstd_control[0], std_control[0], x.float()[0]

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return np.shape(self.data)[0]
