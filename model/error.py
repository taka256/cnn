import numpy as np

class Error:

    class squared(object):

        def __init__(self):
            'squared class'

        def delta(self, y, t):
            return (y - t).T.dot(y - t).sum() / 2.0

        def derivated_delta(self, y, t):
            return y - t


    class cross_entropy(object):

        def __init__(self):
            'cross entropy'

        def delta(self, y, t):
            return (-t * np.log(y)).sum()

        def derivated_delta(self, y, t):
            return (y - t) / (y * (1.0 - y))
