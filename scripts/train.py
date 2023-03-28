import sys
sys.path.append("/home/lherron/scratch/DDIM/thermodynamic-diffusion/thermodiff")
from Backbone import *
from DiffusionModel import *
from DiffusionProcesses import *
from Loader import *
from utils import *
from architectures.UNet2D import Unet2D
import argparse
print("modules imported")

parser = argparse.ArgumentParser()
parser.add_argument('--pdbid', required=True,
                    type=str, help="pdbid is required")
args = parser.parse_args()

# pdbid is required input
pdb=args.pdbid

# initializing directory
device = "cuda" if torch.cuda.is_available() else "cpu"
num_devices = torch.cuda.device_count()
sys_yaml_path = "/home/lherron/scratch/DDIM/sys_config.yaml"
exp_yaml_path = "/home/lherron/scratch/DDIM/exp_config.yaml"
directory = Directory(pdb, sys_yaml_path, exp_yaml_path, device, num_devices)

# intializing data loader
loader = Loader(directory,
                control_tuple = (1,-1),
                transform_type = "whiten"
               )

# initializing model architecture
model_dim = compute_model_dim(loader.data_dim, groups=8)

model = Unet2D(dim = model_dim,
               dim_mults = (1,2,4),
               resnet_block_groups = 8,
               learned_variance = False,
               self_condition = True,
               learned_sinusoidal_cond = True,
               channels=5
              )

# initializing backbone from architecture
backbone = ConvBackbone(model=model,
                        data_shape=loader.get_data_dim(),
                        target_shape=model_dim,
                        num_dims=4,
                        lr=1e-3
                       )

# intializing diffusion process
diffusion = VPDiffusion(num_diffusion_timesteps=100)

# initializing trainer
trainer = DiffusionTrainer(diffusion,
                           backbone,
                           loader,
                           directory,
                           "noise"
                          )

#training
trainer.train(100, loss_type="l2", batch_size=64, print_freq=1)
