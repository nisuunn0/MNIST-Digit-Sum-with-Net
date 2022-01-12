import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from mnist import MNIST # for loading the MNIST dataset
import numpy as np
from sklearn.utils import shuffle # for shuffleing dataset prior to converting the dataset from numpy to pytorch tensor
from tqdm import tqdm # for any loop progress displaying
import sys
import network1 as net1 # network defined in network1.py
import torch.nn as nn

######################################## Data loading and processing ########################################
np.set_printoptions(threshold = sys.maxsize) # have this active if you wish to fully print an np array in the terminal

# load training data
mndata = MNIST('./mnistData', mode = 'randomly_binarized') # randomly binarized will make sure that the dataset is binary
images, labels = mndata.load_training() # 60000 samples for training
# load testing data 
images_test, labels_test = mndata.load_testing() # 10000 samples

# shuffle both the training data and the testing data, prior to concatenating concatenating the images in groups of 3
np_images = np.array(images) # convert to numpy
np_labels = np.array(labels)
np_images, np_labels = shuffle(np_images, np_labels)

np_images_test = np.array(images_test)
np_labels_test = np.array(labels_test)
np_images_test, np_labels_test = shuffle(np_images_test, np_labels_test)

# concatenate the training images and their respective labels, also for testing data
#conc_images = np.zeros(shape = (20000, 2352)) # will store 3 concatenated images. 1 image is 784 values, so 3 images is 3 * 784 = 2352 values
conc_labels = np.zeros(shape = 20000) # wil store the sum of the above 3 concatenated images
conc_images = np.zeros(shape = (20000, 28, 84)) # same as above but use this if you want the image to be 2d 28*28 instead of 1d 784


#conc_images_test = np.zeros(shape = (3333, 2352)) # will store 3 concatenated images, 1 image will be left out as 10000 isn't divisible by 3 (testing set has 10000 entries)
conc_images_test = np.zeros(shape = (3333, 28, 84)) # same as above but if the images are to be stored in 2d 28*28 format instead of 1d 784
conc_labels_test = np.zeros(shape = 3333) # will store the sum of the above 3 concatenated images, 1 label will be left out as 10000 isn't divisible by 3

# The training images will be concatenated horizontally
# fill up the above arrays:
counter = 0 # index for conc_images and conc_labels
counter1 = 0 # index for conc_images_test and conc_abels_test
for i in tqdm(range(0, len(np_labels), 3)):
    """
    # original, flattened, without reshaping into 28*28, for 1d 784
    i1 = np_images[i]
    i2 = np_images[i + 1]
    i3 = np_images[i + 2]
    l1 = np_labels[i]
    l2 = np_labels[i + 1]
    l3 = np_labels[i + 2]
    """
    # use this if you wanna reshape each of the images into 2d 28*28 instead of 1d 784
    # gather the next 3 images and their respective labels, then concatenate the 3 images horizontally, and sum their 3 labels into one label.
    i1 = np_images[i].reshape((28, 28))
    i2 = np_images[i + 1].reshape((28, 28))
    i3 = np_images[i + 2].reshape((28, 28))
    l1 = np_labels[i]
    l2 = np_labels[i + 1]
    l3 = np_labels[i + 2]

    # concatenate i1, i2 and i3
    c1 = np.hstack((i1, i2))
    c2 = np.hstack((c1, i3))

    # sum l1, l2 and l3
    l_sum = l1 + l2 + l3

    # put c2 and l_sum into conc_images and conc_labels respectively
    conc_images[counter] = c2
    conc_labels[counter] = l_sum
    counter += 1
    
    # same as above but for the test data, which has less elements than training data
    if (i < (len(np_labels_test) - 3)):
        """
        # original, flattened, without reshaping into 28*28, for 1d 784
        im1 = np_images_test[i]
        im2 = np_images_test[i + 1]
        im3 = np_images_test[i + 2]
        """

        # same as above but if the images are 2d 28*28 rather than 1d 784
        im1 = np_images_test[i].reshape((28, 28))
        im2 = np_images_test[i + 1].reshape((28, 28))
        im3 = np_images_test[i + 2].reshape((28, 28))

        la1 = np_labels_test[i]
        la2 = np_labels_test[i + 1]
        la3 = np_labels_test[i + 2]

        # concatenate im1, im2 and im3
        co1 = np.hstack((im1, im2))
        co2 = np.hstack((co1, im3))

        # sum la1, la2, la3
        la_sum = la1 + la2 + la3

        # put co2 and the la_sum into conc_images_test and conc_labels_test respectively
        conc_images_test[counter1] = co2
        conc_labels_test[counter1] = la_sum
        counter1 += 1

# Load numpy array data to pytorch
#print('sample from concatenated and reshaped images: ', conc_images[0])
#np.set_printoptions(threshold = np.inf, linewidth = np.inf)
#with open('arr1.txt', 'w') as f:
#    f.write(np.array2string(conc_images[0], separator = ', '))
#print('corresponding label: ', conc_labels[0])

# transform np array to torch tensor, for both the training and test data
tensor_x = torch.from_numpy(conc_images).float() # type change added to fit pytorch required format
tensor_x = tensor_x.unsqueeze(1) # So that the tensor_x input fits the format requested by pytorch (adds one more dimension to the input)
tensor_y = torch.from_numpy(conc_labels).long() # type change added to fit pytorch required format

tensor_x_test = torch.from_numpy(conc_images_test).float()
tensor_x_test = tensor_x_test.unsqueeze(1)
tensor_y_test = torch.from_numpy(conc_labels_test).long() 

# some parameters 
num_epochs = 5
num_classes = 28 # 0-27
batch_size = 100
learning_rate = 0.001
model_path = './model1' # path to save model in

########################### set up datasets and dataloaders for both the training and the testing sets. ###########################################
dataset = TensorDataset(tensor_x, tensor_y) # create dataset
dataset_test = TensorDataset(tensor_x_test, tensor_y_test)
data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True) # create dataloader, wrap an iterable around the dataset.
data_loader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle = False)

# print shapes of data
for X, y in data_loader:
    print('shape of X: ', X.shape)
    print('shape of y: ', y.shape)
    break
for X, y in data_loader_test:
    print('shape of X train: ', X.shape)
    print('shape of y train: ', y.shape)
    break

# setup for training the network
net = net1.network1()
print('net: ', net)

criterion = nn.CrossEntropyLoss() # Loss
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate) # optimizer

###################################### begin training the model #########################################################
total_steps = len(data_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(data_loader):
        # forward pass
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # backpropagate and apply Adam's optmization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], step [{}/{}], loss: {:.4f}, accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, total_steps, loss.item(), (correct / total) * 100))

#################################### Test the model ###############################
net.eval()
with torch.no_grad():
    n_correct = 0
    total = 0
    for images, labels in data_loader_test:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    print('Accuracy during testing of the model on 10000 images: {} %'.format((n_correct / total) * 100))

# save the model
torch.save(net.state_dict(), model_path + 'net_model.ckpt')
