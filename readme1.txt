An attempt at predicting MNIST digit's sums. This approach concatenates 3 MNIST images horizontally and sums their respective labels to get one label.
network1.py contains code for the neural network
runner2.py has code which prepares and preprocesses the MNIST data, sets up the training and testing of a neural network using network1.py.
runner2.py expects the MNIST dataset to be available in a folder: './mnistData'.
python libraries used: torch (pytorch), torchvision, mnist (python MNIST dataset loading library), numpy, sklearn, tqdm, and sys. 

Highest observed accuracy was around 32% with 5 epochs.
