import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
from PIL import Image
import h5py

from glob import glob
import argparse
from time import time

from data import make_data_loader

device = torch.device('cuda:0')

parser = argparse.ArgumentParser()
parser.add_argument('--data_adress', type=str, default='./test1/',
                    help='data address')
parser.add_argument('--model_weights', type=str, default='./tradeoff_network_vgg3_case10_03-0.98.hdf5',
                    help='data address')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Total batch size')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Data loading num_workers')
parser.add_argument('--gpu_mode', type=int, default=1,
                    help='Gpu mode, 0: Single, 1: Multi, 2: Both and compare')

args = parser.parse_args()

# NOTE:
# Don't know how to do Multigpu, is it possible? Needs testing
# Not sure if model loading is correct


# TODO:
# Add transpose to args option, since not sure about kernel
# random state?

# Function to check total number of parameters
def check_total_params(network):
    params = network.parameters()
    total = 0

    for param in params:
        current = 0
        sizes = param.size()
        if  len(sizes) == 4:
            current = sizes[0] * sizes[1] * sizes[2] * sizes[3]
        elif len(sizes) == 2:
            current = sizes[0] * sizes[1]
        elif len(sizes) == 1:
            current = sizes[0]
        else:
            print(sizes)
            print("error")
        total += current
    print(total)

# Model Definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Block 1
        self.block1_conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.block1_conv2 = nn.Conv2d(64, 64, 3, 1, 1)

        # Block 2
        self.block2_conv1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.block2_conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        
        # Block 3
        self.block3_conv1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.block3_conv2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.block3_conv3 = nn.Conv2d(256, 256, 3, 1, 1)
        
        # Block 4
        self.block4_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.block4_conv2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.block4_conv3 = nn.Conv2d(512, 512, 3, 1, 1)
        
        # Block 5
        self.block5_conv1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.block5_conv2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.block5_conv3 = nn.Conv2d(512, 512, 3, 1, 1)

        self.fc = nn.Linear(25088, 3)

    def forward(self, x):
        # Block 1
        x = F.relu(self.block1_conv1(x))
        x = F.max_pool2d(F.relu(self.block1_conv2(x)), (2, 2))

        # Block 2
        x = F.relu(self.block2_conv1(x))
        x = F.max_pool2d(F.relu(self.block2_conv2(x)), (2, 2))
        
        # Block 3
        x = F.relu(self.block3_conv1(x))
        x = F.relu(self.block3_conv2(x))
        x = F.max_pool2d(F.relu(self.block3_conv3(x)), (2, 2))
        
        # Block 4
        x = F.relu(self.block4_conv1(x))
        x = F.relu(self.block4_conv2(x))
        x = F.max_pool2d(F.relu(self.block4_conv3(x)), (2, 2))
        
        # Block 5
        x = F.relu(self.block5_conv1(x))
        x = F.relu(self.block5_conv2(x))
        x = F.max_pool2d(F.relu(self.block5_conv3(x)), (2, 2))

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

# Notes:

# given format: 3,3,3,64
# torch format: 64,3,3,3

# Better example:
# given format: 3,3,64,128
# torch format: 128,64,3,3

# convert kernelxkernel , incoming, outgoing TO outgoing, incoming, kernelxkernel

def make_param(pytorch_tensor, bias=False):
    # NOTE: Experimental, one should work, not sure if kernel weight should be transposed or not.
    if not bias:
        pytorch_tensor = pytorch_tensor.transpose((3,2,0,1))
        # pytorch_tensor = h5_tensor.transpose((3,2,1,0))

    # Convert to parameter
    pytorch_tensor = torch.nn.Parameter(torch.from_numpy(pytorch_tensor))
    return pytorch_tensor

# Loading weights to model
f = h5py.File(args.model_weights)
blocked_layers = [[net.block1_conv1, net.block1_conv2], [net.block2_conv1, net.block2_conv2],
                [net.block3_conv1, net.block3_conv2, net.block3_conv3], [net.block4_conv1,
                 net.block4_conv2, net.block4_conv3], [net.block5_conv1, net.block5_conv2, 
                 net.block5_conv3]]

for block_number, block in enumerate(blocked_layers):
    block_number += 1
    for layer_number, layer in enumerate(block):
        layer_number += 1
        address = "block{}_conv{}".format(block_number, layer_number)
        # print(layer)
        layer.weight = make_param(f['/model_weights/{}/'.format(address)][address]['kernel:0'][:])
        layer.bias = make_param(f['/model_weights/{}/'.format(address)][address]['bias:0'][:], True)

    # TODO: add f['/model_weights/dense']['dense']['kernel:0'] and bias for last layer

net = net.to(device)

# Data loading

data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


loader = make_data_loader(args.data_adress, transforms=data_transforms,
                batch_size=args.batch_size, num_workers=args.num_workers)

if args.gpu_mode == 0 or args.gpu_mode == 2:
    tim = time()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = net.forward(images)

        print(outputs.shape)
        
    print("Time taken to infer with single gpu: ", time() - tim)

if args.gpu_mode == 1 or args.gpu_mode == 2:
    
    net = nn.DataParallel(net)
    tim = time()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = net.forward(images)

        print(outputs.shape)
        
    print("Time taken to infer with multiple gpus: ", time() - tim)



