import os

import pandas as pd
import sys
from keras import Sequential
from keras.layers import Dense
from keras.models import load_model


def read_datasets(dir):
    x = pd.read_csv(dir + "global_stat_input.csv")
    y = pd.read_csv(dir + "global_stat_output.csv")

    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    x = x.fillna(1)
    y = y.fillna(1)
    return x, y


if __name__ == '__main__':
    os.chdir('../../..')  # going to the root of the project

    if len(sys.argv) < 2:
        raise Exception("action required!")

    action = sys.argv[1]
    if action == 'train':
        x_train, y_train = read_datasets("supplementary/data/training/")

        number_of_features = x_train.columns.size
        number_of_outputs = y_train.columns.size
        number_of_layers = 4
        number_of_units = [16, 32, 16, number_of_outputs]
        activations = ['relu', 'relu', 'relu', 'linear']

        model = Sequential()
        for i in range(number_of_layers):
            model.add(Dense(number_of_units[i], activation=activations[i]))

        model.compile(optimizer='sgd', loss='mse', metrics=['mse', 'mae'])
        model.fit(x_train.to_numpy(), y_train.to_numpy(), epochs=150)

        if len(sys.argv) > 2 and sys.argv[2] == '--save':
            model.save('supplementary/models/single/single_nn.h5')

    elif action == 'evaluate':
        model = load_model('supplementary/models/single/single_nn.h5')

        x_test, y_test = read_datasets("target/data/")
        print(model.evaluate(x_test.to_numpy(), y_test.to_numpy()))
