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

#Check the accuracy of our model
#my_network.load_weights('meals.h5')
#print(my_network.predict(np.array([
   # [1,0,0,0,0,1]
#])))


#free up resources
keras.backend.clear_session()

