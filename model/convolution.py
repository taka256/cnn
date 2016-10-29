import numpy as np

class Convolution(object):

    def __init__(self, m, k, kh, kw, act):
        self.kh = kh
        self.kw = kw
        self.weight = np.random.randn(m, k, kh, kw) * 0.1
        self.bias = np.random.randn(m) * 0.1
        self.activator = act


    def forward(self, X):
        return self.activator.activate(self.__forward(X))


    def backward(self, delta, shape):
        k, h, w = delta.shape
        delta_patch = np.tensordot(delta.reshape(k, h * w), self.weight, (0, 0))
        return self.__patch2im(delta_patch, h, w, shape)


    def update_weight(self, delta, epsilon):
        k, h, w = delta.shape
        self.weight -= epsilon * np.tensordot(delta.reshape(k, h * w), self.__patch, (1, 0))
        # self.bias -= epsilon * delta.reshape(k, h * w).sum(axis = 1)


    def __forward(self, X):
        xh, xw = X.shape[1:3]
        m = self.weight.shape[0]
        oh, ow = xh - self.kh / 2 * 2, xw - self.kw / 2 * 2
        self.__patch = self.__im2patch(X, oh, ow)
        return np.tensordot(self.__patch, self.weight, ((1, 2, 3), (1, 2, 3))).T.reshape(m, oh, ow)# + self.bias.reshape(m, 1, 1)


    def __patch_center(self, h, w):
        _l = np.arange(h * w)
        return np.vstack((_l / w, _l % w)).T
        

    def __im2patch(self, X, oh, ow):
        c = self.__patch_center(oh, ow)
        return np.array([X[:, j:j+self.kh, i:i+self.kw] for j, i in c])


    def __patch2im(self, patch, h, w, shape):
        im = np.zeros(shape)
        c = self.__patch_center(h, w)
        for j, i in c:
            im[:, j:j+self.kh, i:i+self.kw] += patch[j * w + i]
        return im
