import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
from PIL import Image

# Export to TorchScript that can be used for LibTorch

torch.backends.cudnn.benchmark = True

# From cuLANE, Change this line if you are using TuSimple
cls_num_per_lane = 18
griding_num = 200
backbone =18

net = parsingNet(pretrained = False,backbone='18', cls_dim = (griding_num+1,cls_num_per_lane,4),
                use_aux=False)

# Change test_model where your model stored.
test_model = '/data/Models/UltraFastLaneDetection/culane_18.pth'

#state_dict = torch.load(test_model, map_location='cpu')['model'] # CPU
state_dict = torch.load(test_model, map_location='cuda')['model'] # CUDA
compatible_state_dict = {}
for k, v in state_dict.items():
    if 'module.' in k:
        compatible_state_dict[k[7:]] = v
    else:
        compatible_state_dict[k] = v

net.load_state_dict(compatible_state_dict, strict=False)
net.eval()

# Test Input Image
img = torch.zeros(1, 3, 288, 800)  # image size(1,3,320,192) iDetection
y = net(img)  # dry run

ts = torch.jit.trace(net, img)

#ts.save('UFLD.torchscript-cpu.pt') # CPU
ts.save('UFLD.torchscript-cuda.pt') # CUDA