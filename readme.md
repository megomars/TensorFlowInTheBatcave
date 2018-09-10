# TensorFlow in the Batcave
[Tensorflow course](https://code.tutsplus.com/courses/learn-machine-learning-with-google-tensorflow/lessons/setting-up-tensorflow)

## Setup
The first step is to install a virtual environment on your machine

```Python
pip install virtualenv
virtualenv my_tf_environment # create a virtual environment
cd my_tf_environment 
source bin/activate # You must activate the environment every time you open a new terminal window.

# Now we can install Tensorflow - other dependencies such as numpy are installed
pip install tensorflow
# save and restore tensorflow neural networks with the h5py module
pip install h5py
# easier to read and manage data
pip install pandas
```

## Neural networks
![Neural networks](nn.png)

There are three kinds of layers in a neural network. The input layer, the hidden layers and the output layer. We will be using Tensorflow's Keras package to build our NN.


## Example 1
The Python code for this looks as follows:

```Python
from tensorflow import keras
import numpy as np

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
my_network.compile(
    optimizer='adam',
    loss='mse'
)
#We'll use a numpy array
inputs=np.array([
    [1,0,0,1,0,0],
    [1,0,0,0,1,0],
    [1,0,0,0,0,1],
    [0,1,0,1,0,0],
    [0,1,0,0,1,0],
    [0,1,0,0,0,1],
    [0,0,1,1,0,0],
    [0,0,1,0,1,0],
    [0,0,1,0,0,1]
])
outputs=np.array([
    [0,1,0,0],
    [0,0,1,0],
    [0,1,0,0],
    [1,0,0,0],
    [0,0,1,0],
    [0,0,0,1],
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1]
])

#Train the network
my_network.fit(inputs,outputs,epochs=5000)

#Export a file
my_network.save_weights('meals.h5')

#free up resources
keras.backend.clear_session()
```
Now run the Python file with python network1.py
Next we want to test the accuracy of our model:

```Python
from tensorflow import keras
import numpy as np

my_network = keras.Sequential()
my_network.add(keras.layers.Dense(
    12,
    input_dim=6,
    activation='relu'
))
my_network.add(keras.layers.Dense(
    8,
    activation='relu'
))
my_network.add(keras.layers.Dense(
    4,
    activation='softmax'
))
my_network.compile(
    optimizer='adam',
    loss='mse'
)
inputs=np.array([
    [1,0,0,1,0,0],
    [1,0,0,0,1,0],
    [1,0,0,0,0,1],
    [0,1,0,1,0,0],
    [0,1,0,0,1,0],
    [0,1,0,0,0,1],
    [0,0,1,1,0,0],
    [0,0,1,0,1,0],
    [0,0,1,0,0,1]
])
outputs=np.array([
    [0,1,0,0],
    [0,0,1,0],
    [0,1,0,0],
    [1,0,0,0],
    [0,0,1,0],
    [0,0,0,1],
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1]
])

#Check the accuracy of our model
my_network.load_weights('meals.h5')
print(my_network.predict(np.array([
    [1,0,0,0,0,1]
])))

#free up resources
keras.backend.clear_session()
```
First run the following script:
python meals_train.py

Next run the following script:
python meals_validate.py
![result from Python network1.py](mealtest.png)

## Example 2
Next we will train a NN to classify zoo animals based on the following dataset: [Zoo animal classification dataset](https://www.kaggle.com/uciml/zoo-animal-classification)

First run the following script:
python animals_train.py

Next run the following script:
python animals_validate.py

## Example 3
Convolutional NN for image classification based on the following dataset: [Fruits 360 Dataset](https://www.kaggle.com/moltean/fruits)

First run the following script:
python fruits_train.py

Next run the following script:
python fruits_validate.py
![Image classification neural network](example3.png)