from tensorflow import keras

#Multi layer perceptrons MLPs
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