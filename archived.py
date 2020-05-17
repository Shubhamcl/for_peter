# NOTE: Better way to do this would be state dictionary, but meh
# Loading weights:

# # Block 1
# net.block1_conv1.weight = make_param(f['/model_weights/block1_conv1/']['block1_conv1']['kernel:0'][:])
# net.block1_conv1.bias = make_param(f['/model_weights/block1_conv1/']['block1_conv1']['bias:0'][:], bias=True)

# net.block1_conv2.weight = make_param(f['/model_weights/block1_conv2/']['block1_conv2']['kernel:0'][:])
# net.block1_conv2.bias = make_param(f['/model_weights/block1_conv2/']['block1_conv2']['bias:0'][:], bias=True)

# # Block 2
# net.block2_conv1.weight = make_param(f['/model_weights/block2_conv1/']['block2_conv1']['kernel:0'][:])
# net.block2_conv1.bias = make_param(f['/model_weights/block2_conv1/']['block2_conv1']['bias:0'][:], bias=True)

# net.block2_conv2.weight = make_param(f['/model_weights/block2_conv2/']['block2_conv2']['kernel:0'][:])
# net.block2_conv2.bias = make_param(f['/model_weights/block2_conv2/']['block2_conv2']['bias:0'][:], bias=True)

# # Block 3
# net.block3_conv1.weight = make_param(f['/model_weights/block3_conv1/']['block3_conv1']['kernel:0'][:])
# net.block3_conv1.bias = make_param(f['/model_weights/block3_conv1/']['block3_conv1']['bias:0'][:], bias=True)

# net.block3_conv2.weight = make_param(f['/model_weights/block3_conv2/']['block3_conv2']['kernel:0'][:])
# net.block3_conv2.bias = make_param(f['/model_weights/block3_conv2/']['block3_conv2']['bias:0'][:], bias=True)

# net.block3_conv3.weight = make_param(f['/model_weights/block3_conv3/']['block3_conv3']['kernel:0'][:])
# net.block3_conv3.bias = make_param(f['/model_weights/block3_conv3/']['block3_conv3']['bias:0'][:], bias=True)

# # Block 4
# net.block4_conv1.weight = make_param(f['/model_weights/block3_conv1/']['block3_conv1']['kernel:0'][:])
# net.block4_conv1.bias = make_param(f['/model_weights/block3_conv1/']['block3_conv1']['bias:0'][:], bias=True)

# net.block4_conv2.weight = make_param(f['/model_weights/block3_conv2/']['block3_conv2']['kernel:0'][:])
# net.block4_conv2.bias = make_param(f['/model_weights/block3_conv2/']['block3_conv2']['bias:0'][:], bias=True)

# net.block4_conv3.weight = make_param(f['/model_weights/block3_conv3/']['block3_conv3']['kernel:0'][:])
# net.block4_conv3.bias = make_param(f['/model_weights/block3_conv3/']['block3_conv3']['bias:0'][:], bias=True)


# layers = []



# depth_of_two = [[net.block1_conv1, net.block1_conv2], [net.block2_conv1, net.block2_conv2]]
# depth_of_three = [net.block3_conv1, net.block3_conv2, net.block3_conv3, net.block4_conv1,
                #  net.block4_conv2, net.block4_conv3, net.block5_conv1, net.block5_conv2, 
                #  net.block5_conv3]


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



print(net.block2_conv1.bias.shape)
print(net.block2_conv1.weight.shape)
print(f['/model_weights/block2_conv1/']['block2_conv1']['kernel:0'][:].shape)

cat = Image.open('/home/shubham/Desktop/git/for_peter/train/cat.1.jpg')
tensor = transforms.ToTensor()(cat.resize((360,360))).unsqueeze(0)
print(net.forward(tensor).shape)