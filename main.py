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
