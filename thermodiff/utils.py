import os
import yaml
import torch.nn.functional as F
import numpy as np

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if is_lambda(d) else d

def load_yaml(file):
    with open(file, 'r') as stream:
            yaml_loaded = yaml.safe_load(stream)
    return yaml_loaded

def compute_model_dim(data_dim, groups):
    return int(np.ceil(data_dim / groups) * groups)

class Interpolater: # belongs in utils
    """
    Reshapes irregularly (or unconventionally) shaped data to be compatible with a model
    """
    def __init__(self,
                 data_shape: tuple,
                 target_shape: tuple):
        self.data_shape, self.target_shape = data_shape, target_shape

    def to_target(self, x):
        return F.interpolate(x, size=self.target_shape, mode='nearest-exact')

    def from_target(self, x):
        return F.interpolate(x, size=self.data_shape, mode='nearest-exact')

class Directory: # belongs in utils
    """
    Reads relevant paths from a yaml file defined for an experiment.
    """
    def __init__(self,
                 pdb: str,
                 sys_yaml_path: str,
                 exp_yaml_path: str,
                 expid: str,
                 device: str,
                 num_devices: int
                ):
        self.device_ids = list(range(0,num_devices))
        # loading yaml files
        self.system_params = load_yaml(sys_yaml_path)
        self.exp_params = load_yaml(exp_yaml_path)

        # replacing wildcards in loaded yaml files
        self.pdb = pdb
        wildcards = {"PDBID" : self.pdb,
                     "EXPID" : expid}
        self.exp_params = self.replace_wildcards(self.exp_params, wildcards)
        print(self.exp_params)

        self.identifier = self.exp_params["paths"]["identifier"]
        self.base_path = self.system_params["system"]["exp_path"]
        self.model_path = os.path.join(self.base_path, self.exp_params["paths"]["model_path"])
        self.data_path = os.path.join(self.base_path, self.exp_params["paths"]["dataset_path"])
        self.sample_path = os.path.join(self.base_path, self.exp_params["paths"]["sample_path"])
        self.device = device

    @staticmethod
    def replace_wildcards(d, wildcard_d):
        for k_header, d_ in d.items():
            for k, v in d_.items():
                if isinstance(v, str):
                    for k_, v_ in wildcard_d.items():
                        v = v.replace(k_, v_)
                    d[k_header][k] = v
        return d

    def get_backbone_path(self):
        return self.model_path

    def get_dataset_path(self):
        return self.data_path

    def get_sample_path(self):
        return self.sample_path

    def get_device(self):
        return self.device

    def get_pdb(self):
        return self.pdb
