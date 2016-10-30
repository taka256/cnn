import numpy as np

class Convolution(object):

    def __init__(self, m, k, kh, kw, act):
        self.kh = kh
        self.kw = kw
        self.weight = np.random.randn(m, k, kh, kw) * 0.1
        self.activator = act


    def forward(self, X):
        return self.activator.activate(self.__forward(X))


    def backward(self, delta, shape):
        s_batch, k, h, w = delta.shape
        delta_patch = np.tensordot(delta.reshape(s_batch, k, h * w), self.weight, (1, 0))
        return self.__patch2im(delta_patch, h, w, shape)


    def update_weight(self, delta, epsilon):
        s_batch, k, h, w = delta.shape
        self.weight -= epsilon * self.__grad(delta, s_batch, k, h, w)


    def __forward(self, X):
        s_batch, k, xh, xw = X.shape
        m = self.weight.shape[0]
        oh, ow = xh - self.kh / 2 * 2, xw - self.kw / 2 * 2
        self.__patch = self.__im2patch(X, s_batch, k, oh, ow)
        return np.tensordot(self.__patch, self.weight, ((2, 3, 4), (1, 2, 3))).swapaxes(1, 2).reshape(s_batch, m, oh, ow)


    def __im2patch(self, X, s_batch, k, oh, ow):
        patch = np.zeros((s_batch, oh * ow, k, self.kh, self.kw))
        for j in range(oh):
            for i in range(ow):
                patch[:, j * ow + i, :, :, :] = X[:, :, j:j+self.kh, i:i+self.kw]
        return patch


    def __patch2im(self, patch, h, w, shape):
        im = np.zeros(shape)
        for j in range(h):
            for i in range(w):
                im[:, :, j:j+self.kh, i:i+self.kw] += patch[:, j * w + i]
        return im


    def __grad(self, delta, s_batch, k, h, w):
        return np.tensordot(delta.reshape(s_batch, k, h * w), self.__patch, ((0, 2), (0, 1))) / s_batch
