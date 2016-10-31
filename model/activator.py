import numpy as np

class Activator:

    class linear(object):

        def __init__(self):
            'linear class'

        def activate(self, x):
            return x

        def derivate(self, x):
            return np.ones_like(x)


    class sigmoid(object):

        def __init__(self):
            'sigmoid class'

        def activate(self, x):
            return 1.0 / (1.0 + np.exp(-x))

        def derivate(self, x):
            return x * (1.0 - x)


    class softmax(object):

        def __init__(self):
            'softmax class'

        def activate(self, x):
            exp = np.exp(x)
            return exp / exp.sum(axis = 0)

        def derivate(self, x):
            return x * (1.0 - x)


    class relu(object):

        def __init__(self):
            'relu class'

        def activate(self, x):
            return np.maximum(x, 0.0)

        def derivate(self, x):
            return np.vectorize(lambda _x: 0.0 if _x < 0.0 else 1.0)(x)
