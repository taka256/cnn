import numpy as np
import os
from model.cnn import CNN
from model.nn import NN
from model.convolution import Convolution as conv
from model.pooling import Pooling as pool
from model.activator import Activator as act
from model.error import Error as err

def read_data(fn):
    ml = np.loadtxt(fn, delimiter = ',')
    X, t = np.hsplit(ml, [-1])
    return (X / X.max(), t.astype('int'))

def create_label(t, n_data, n_class):
    T = np.zeros((n_data, n_class))
    T[np.arange(n_data), t[:, 0]] = 1.0
    return T

if __name__ == '__main__':

    print 'read data...'
    fn = '{0}/mldata/mnist_train_data.csv'.format(os.getenv('DPATH')[:-1])
    X, t = read_data(fn)
    n_data, n_input = X.shape
    n_class = np.unique(t).size
    T = create_label(t, n_data, n_class)

    print 'make train/test data'
    n_train, n_test = 1000, 50
    i = np.random.permutation(n_data)[:n_train+n_test]
    i_train, i_test = np.hsplit(i, [n_train])
    X_train, X_test = X[i_train, :].reshape(n_train, 1, 28, 28), X[i_test, :].reshape(n_test, 1, 28, 28)
    T_train, T_test = T[i_train, :], T[i_test, :]

    print 'initialize...'
    linear, sigmoid, softmax, relu = act.linear(), act.sigmoid(), act.softmax(), act.relu()
    conv1, conv2 = conv(20, 1, 5, 5, relu), conv(50, 20, 5, 5, relu)
    pool1, pool2 = pool(2, 2, 2), pool(2, 2, 2)
    neural = NN(800, 500, 10, linear, sigmoid, softmax)
    error = err.cross_entropy()
    cnn = CNN(conv1, pool1, conv2, pool2, neural, error)

    print 'train...'
    cnn.train(X_train, T_train, epsilon = 0.005, lam = 0.0001, gamma = 0.9, s_batch = 1, epochs = 50)

    # print 'predict...'
    # Y_test = cnn.predict(X_test)
    # print 'test loss: {0}'.format(cnn.test_loss(Y_test, T_test))

    print 'save figure of loss...'
    cnn.save_lossfig()
