import numpy as np

class Pooling(object):

    def __init__(self, kh, kw, s):
        self.kh = kh
        self.kw = kw
        self.s = s


    def forward(self, X):
        k, h, w = X.shape
        kh, kw = self.kh, self.kw
        oh, ow = np.vectorize(lambda _x: _x / self.s + 1)([h - kh, w - kw])
        val, self.__ind = self.__max(X, k, oh, ow)
        return val


    def backward(self, X, delta, act):
        k, h, w = X.shape
        oh, ow = delta.shape[1:]
        r = (h / oh) * (w / ow)
        ind = np.arange(k * oh * ow) * r + self.__ind.flatten()
        return self.__backward_delta(delta, ind, k, h, w) * act.derivate(X)


    def __max(self, X, k, oh, ow):
        patch = self.__im2patch(X, k, oh, ow)
        return map(lambda _f: _f(patch, axis = 2).reshape(k, oh, ow), [np.max, np.argmax])


    def __im2patch(self, X, k, oh, ow):
        l = self.__patch_tl(oh, ow)
        return np.array([X[:, j:j+self.kh, i:i+self.kw] for j, i in l]).swapaxes(0, 1).reshape(k, oh * ow, -1)


    def __patch_tl(self, oh, ow):
        _l = np.arange(oh * ow)
        return np.vstack((_l / ow, _l % ow)).T * self.s
        

    def __backward_delta(self, delta, ind, k, h, w):
        _delta = np.zeros(k * h * w)
        _delta[ind] = delta.flatten()
        return _delta.reshape(k, h, w)
