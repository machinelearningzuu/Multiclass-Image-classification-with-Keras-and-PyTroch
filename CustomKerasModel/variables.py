classes = [ 'paper','rock', 'scissors']
batch_size = 10
valid_size = 4
color_mode = 'rgb'
width = 150
height = 150
target_size = (width, height)
shear_range = 0.2
zoom_range = 0.2
rescale = 1./255
num_classes = 3
epochs = 15
train_step = 252
val_step = 9
test_step = 38
verbose = 2
dense = 64

input_shape = (width, height, 3)
kernal_size = (3, 3)
pool_size = (2, 2)
ofm = 32 # output feature map 1

# data directories and model paths
train_dir = 'data/train/'
test_dir = 'data/test/'
valid_dir = 'data/validation/'
model_architecture = "KerasModel/model.json"
model_weights = "KerasModel/model.h5"
