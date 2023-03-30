import sys
sys.path.append("/home/lherron/scratch/repos/thermodynamic-diffusion/thermodiff")
from Backbone import *
from DiffusionModel import *
from DiffusionProcesses import *
from Loader import *
from utils import *
from UNet2D_mid_attn import Unet2D
import argparse
print("modules imported")

parser = argparse.ArgumentParser()
parser.add_argument('--pdbid', required=True,
                    type=str, help="pdbid is required")
parser.add_argument('--expid', required=True,
                    type=str, help="pdbid is required")
parser.add_argument('--pred_type', required=True,
                    type=str, help="pdbid is required")
parser.add_argument('--self_condition', required=True,
                    type=str, help="pdbid is required")
parser.add_argument('--sys_config_path', required=True,
                    type=str, help="pdbid is required")
parser.add_argument('--exp_config_path', required=True,
                    type=str, help="pdbid is required")
args = parser.parse_args()

# pdbid is required input
pdb=args.pdbid
expid=args.expid
pred_type=args.pred_type
sys_config_path=args.sys_config_path
exp_config_path=args.exp_config_path
self_condition=eval(args.self_condition)

# initializing directory
device = "cuda" if torch.cuda.is_available() else "cpu"
num_devices = torch.cuda.device_count()
directory = Directory(pdb,
                      sys_config_path,
                      exp_config_path,
                      expid,
                      device,
                      num_devices)

# intializing data loader
loader = Loader(directory,
                control_tuple = (1,-1),
                transform_type = "whiten",
                dequantize_scale=1e-3
               )

# initializing model architecture
model_dim = compute_model_dim(loader.data_dim, groups=8)

model = Unet2D(dim = model_dim,
               dim_mults = (1,2,4),
               resnet_block_groups = 8,
               learned_variance = False,
               self_condition = self_condition,
               learned_sinusoidal_cond = True,
               channels=5
              )

# initializing backbone from architecture
backbone = ConvBackbone(model=model,
                        data_shape=loader.get_data_dim(),
                        target_shape=model_dim,
                        num_dims=4,
                        lr=1e-3,
                        eval_mode='train',
                        self_condition=self_condition
                       )

# intializing diffusion process
diffusion = VPDiffusion(num_diffusion_timesteps=100)

# initializing trainer
trainer = DiffusionTrainer(diffusion,
                           backbone,
                           loader,
                           directory,
                           pred_type
                          )

#training
trainer.train(100, loss_type="l2", batch_size=128, print_freq=100)
