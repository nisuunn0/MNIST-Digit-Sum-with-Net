import torch.nn as nn

# Neural network setup from the Deep Learning With Sets and Point Clouds paper:
# all nonlinearities are exponential linear units.
# 4 convolution layers followed by max-pooling.. 
# The convolution layers have respectively 16-32-64-128 output channels and 5 x 5 receptive fields (kernel_size i think). 
# Each pooling, fully connected and set-layer is followed by a 20% dropout. 
# models 3 and 4 use simultaneous dropout
# models 1 and 2 have their convolution layers followed by 2 fully connected layers with 128 hidden units.
# in model 3, after the 1st fully connected layer perform set pooling followed by another dense layer with 128 hidden units. 
# in model 4 the convolution layers are followed by a permutation equivariant layer with 128 output channels, followed by set pooling and a fully connected layer with 128 hidden units.
# for optimization learning rate of 0.0003 with Adam using the default Beta1 = 0.9 and Beta2 = 0.999

# Attempt at following along model 1.
class network1(nn.Module):
    def __init__(self):
        super(network1, self).__init__()
        
        # self.layer1-4 are the first four convolution layers followed by max-pooling
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(inplace = True), 
            nn.MaxPool2d(kernel_size = 2, stride = 2) # convolutional layer followed by max pooling
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
        # self.layer5-6 are the two fully connected layers that follow the first four convolutional layers
        self.layer5 = nn.Sequential(
            nn.Linear(640, 128), # originally 128 * 128
            nn.ReLU(inplace = True)
        )

        self.layer6 = nn.Sequential(
            nn.Linear(128, 28), # 28 outputs because classes are 0 to 27?
            nn.ReLU(inplace = True)
        )

        self.drop_out = nn.Dropout()


    def forward(self, x):
        # pass through the first convolutional layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.size(0), - 1) # flatten
        # pass through the two last fully connected layers
        x = self.drop_out(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
