# TensorFlow in the Batcave
[Tensorflow course](https://code.tutsplus.com/courses/learn-machine-learning-with-google-tensorflow/lessons/setting-up-tensorflow)

## Setup
The first step is to install a virtual environment on your machine

```Python
pip install virtualenv
virtualenv my_tf_environment
cd my_tf_environment
source bin/activate

# Now we can install Tensorflow
pip install tensorflow
# save and restore tensorflow neural networks
pip install h5py
# easier to read and manage data
pip install pandas
```

## Neural networks
![Neural networks](nn.png)

There are three kinds of layers in a neural network. The input layer, the hidden layers and the output layer. We will be using Tensorflow's Keras package to build our NN.

The Python code for this looks as follows:

```Python
from tensorflow import keras

my_network = keras.Sequential()
#This is the first hidden layer and the number of inputs
my_network.add(keras.layers.Dense(
    12,
    input_dim=6,
    activation='relu'
))
#This is the second hidden layer
my_network.add(keras.layers.Dense(
    8,
    activation='relu'
))
#This is the output layer
my_network.add(keras.layers.Dense(
    4,
    activation='softmax'
))
```
