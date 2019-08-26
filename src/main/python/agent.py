import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential


class Agent:

    def __init__(self, hyper_parameters=None):
        if hyper_parameters is None:
            hyper_parameters = {}
        self._model = Sequential()
        units = hyper_parameters.get('number_of_units') or [64] * 3
        activations = hyper_parameters.get('activations') or ['linear'] * 3

        self._model.add(Dense(units[0], input_dim=hyper_parameters.get('features') or 3, activation=activations[0]))
        for i in range(1, hyper_parameters.get('number_of_layers') or 3):
            self._model.add(Dense(units[i] or 64, activation=activations[i]))

        self._model.compile(loss=hyper_parameters.get('loss') or 'mae',
                            optimizer=hyper_parameters.get('optimizer') or 'sgd',
                            metrics=hyper_parameters.get('metrics') or ['mae'])

    def output_tensor(self):
        return self._model.output

    def input_tensor(self):
        return self._model.input

    def train(self, predictors, targets, batch_size=32, epochs=10):
        self._model.fit(predictors, targets,
                        batch_size=batch_size,
                        epochs=epochs)

    def test(self, predictors, targets):
        self._model.evaluate(predictors, targets)

    def predict(self, predictors):
        return self._model.predict(predictors)

    def derivative(self, predictors):
        grads = []
        outputs = self._model.output.shape[1]
        features = self._model.input.shape[1]

        for i in range(outputs):
            gradient = tf.gradients(self._model.output[0][i], self._model.input)[0]
            grads.append(tf.reshape(gradient, (1, features)))

        return tf.keras.backend.get_session().run(tf.concat(grads, axis=0), feed_dict={self._model.input: predictors})

    def cor(self, i, j, predictors_vec, samples=1000):
        predictors = predictors_vec.repeat(samples).reshape(samples, len(predictors_vec))
        predictors[:, i] = np.random.uniform(-1, 1, samples)

        targets = self.predict(predictors)

        return np.cov([predictors[:, i], targets[:, j]])[0, 1] / np.sqrt(
            predictors[:, i].var(ddof=1) * targets[:, j].var(ddof=1))


if __name__ == '__main__':
    agent = Agent({
        'number_of_units': [64, 64, 1]
    })
    train_x = np.random.rand(100, 3)
    train_y = train_x.sum(axis=1).reshape(100, 1)
    # ground_truth = tf.placeholder(tf.float32, (None, 1))
    ground_truth = tf.constant(train_y)
    loss = tf.losses.mean_squared_error(ground_truth, agent._model.output)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    sess = tf.keras.backend.get_session()
    for _ in range(100):
        print(sess.run((train, loss),
                       feed_dict={agent._model.input: train_x}))
