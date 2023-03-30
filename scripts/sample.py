import sys
sys.path.append("/home/lherron/scratch/repos/thermodyanmic-diffusion/thermodiff")
from Backbone import *
from DiffusionModel import *
from DiffusionProcesses import *
from Loader import *
from utils import *
from UNet2D import Unet2D
import argparse
print("modules imported")

parser = argparse.ArgumentParser()
parser.add_argument('--pdbid', required=True,
                    type=str, help="pdbid is required")
parser.add_argument('--epoch', required=True,
                    type=int, help="pdbid is required")
parser.add_argument('--gen_temp', required=True,
                    type=float, help="pdbid is required")
parser.add_argument('--num_samples', required=True,
                    type=int, help="pdbid is required")
args = parser.parse_args()

# pdbid is required input
pdb=args.pdbid
sample_epoch=int(args.epoch)
gen_temp=float(args.gen_temp)
num_samples=int(args.num_samples)

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
model_shape = compute_model_dim(loader.data_dim, groups=8)
model = Unet2D(dim = model_shape,
               dim_mults = (1,2,4),
               resnet_block_groups = 8,
               learned_variance = False,
               self_condition = True,
               learned_sinusoidal_cond = True,
               channels=loader.get_num_channels()
              )

# intializing diffusion process
diffusion = VPDiffusion(num_diffusion_timesteps=100)

# initializing backbone from architecture
backbone = ConvBackbone(model=model,
                        data_shape=loader.get_data_dim(),
                        target_shape=model_shape,
                        num_dims=4,
                        lr=1e-3
                       )

backbone.load_model(directory, sample_epoch)

sampler = DiffusionSampler(diffusion,
                           backbone,
                           loader,
                           directory,
                           "noise"
                          )

sampler.sample_loop(num_samples, 5000, f"{pdb}", gen_temp)
