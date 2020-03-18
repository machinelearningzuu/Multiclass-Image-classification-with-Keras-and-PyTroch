classes = [ 'paper','rock', 'scissors']
batch_size = 10
valid_size = 4
color_mode = 'rgb'
width = 224
height = 224
target_size = (width, height)
shear_range = 0.2
zoom_range = 0.2
rescale = 1./255
num_classes = 3
epochs = 5
train_step = 252
val_step = 9
test_step = 38
verbose = 2

# data directories and model paths
train_dir = 'data/train/'
test_dir = 'data/test/'
valid_dir = 'data/validation/'
model_architecture = "VGG16_TransferLearning/vgg16.json"
model_weights = "VGG16_TransferLearning/vgg16.h5"
