import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

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
        self.block1_conv1 = nn.Conv2d(3, 64, 3, 1)
        self.block1_conv2 = nn.Conv2d(64, 64, 3, 1)

        # Block 2
        self.block2_conv1 = nn.Conv2d(64, 128, 3, 1)
        self.block2_conv2 = nn.Conv2d(128, 128, 3, 1)
        
        # Block 3
        self.block3_conv1 = nn.Conv2d(128, 256, 3, 1)
        self.block3_conv2 = nn.Conv2d(256, 256, 3, 1)
        self.block3_conv3 = nn.Conv2d(256, 256, 3, 1)
        
        # Block 4
        self.block4_conv1 = nn.Conv2d(256, 512, 3, 1)
        self.block4_conv2 = nn.Conv2d(512, 512, 3, 1)
        self.block4_conv3 = nn.Conv2d(512, 512, 3, 1)
        
        # Block 5
        self.block5_conv1 = nn.Conv2d(512, 512, 3, 1)
        self.block5_conv2 = nn.Conv2d(512, 512, 3, 1)
        self.block5_conv3 = nn.Conv2d(512, 512, 3, 1)

        self.fc = nn.Linear(25088, 3)  # 6*6 from image dimension

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
        x = self.fc1(x)

        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Print model
net = Net()
print(net)

# Notes:

# ['model_weights']
#['block1_conv1', 'block1_conv2', 'block1_pool', 
# 'block2_conv1', 'block2_conv2', 'block2_pool', 
# 'block3_conv1', 'block3_conv2', 'block3_conv3', 
# 'block3_pool', 'block4_conv1', 'block4_conv2', 
# 'block4_conv3', 'block4_pool', 'block5_conv1', 
# 'block5_conv2', 'block5_conv3', 'block5_pool', 
# 'dense', 'flatten', 'input_1']

# f['/model_weights/block1_conv1/']['block1_conv1'].keys()

# given format: 3,3,3,64
# torch format: 64,3,3,3

# Better example:
# given format: 3,3,64,128
# torch format: 128,64,3,3

# convert kernelxkernel , incoming, outgoing TO outgoing, incoming, kernelxkernel

def transpose(h5_tensor):
    # NOTE: Experimental, one should work
    pytorch_tensor = h5_tensor.transpose((3,2,0,1))
    # pytorch_tensor = h5_tensor.transpose((3,2,1,0))

    # Convert to parameter
    pytorch_tensor = torch.nn.Parameter(pytorch_tensor)
    return pytorch_tensor

f = h5py.File('/home/shubham/Downloads/tradeoff_network_vgg3_case10_03-0.98.hdf5')

print(net.block2_conv1.bias.shape)
print(net.block2_conv1.weight.shape)
print(f['/model_weights/block2_conv1/']['block2_conv1']['kernel:0'][:].shape)


# NOTE: Better way to do this would be state dictionary, but meh
# Loading weights:

net.block1_conv1.weight = transpose(f['/model_weights/block1_conv1/']['block1_conv1']['kernel:0'][:])
net.block1_conv1.bias = transpose(f['/model_weights/block1_conv1/']['block1_conv1']['bias:0'][:])

net.block1_conv2 = nn.Conv2d(64, 64, 3, 1)

# Block 2
net.block2_conv1 = nn.Conv2d(64, 128, 3, 1)
net.block2_conv2 = nn.Conv2d(128, 128, 3, 1)

# Block 3
self.block3_conv1 = nn.Conv2d(128, 256, 3, 1)
self.block3_conv2 = nn.Conv2d(256, 256, 3, 1)
self.block3_conv3 = nn.Conv2d(256, 256, 3, 1)

# Block 4
self.block4_conv1 = nn.Conv2d(256, 512, 3, 1)
self.block4_conv2 = nn.Conv2d(512, 512, 3, 1)
self.block4_conv3 = nn.Conv2d(512, 512, 3, 1)

# Block 5
self.block5_conv1 = nn.Conv2d(512, 512, 3, 1)
self.block5_conv2 = nn.Conv2d(512, 512, 3, 1)
self.block5_conv3 = nn.Conv2d(512, 512, 3, 1)
