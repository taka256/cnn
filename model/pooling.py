import numpy as np

class Pooling(object):

    def __init__(self, kh, kw, s):
        self.kh = kh
        self.kw = kw
        self.s = s


    def forward(self, X):
        s_batch, k, h, w = X.shape
        oh, ow = (h - self.kh) / self.s + 1, (w - self.kw) / self.s + 1
        val, self.__ind = self.__max(X, s_batch, k, oh, ow)
        return val


    def backward(self, X, delta, act):
        s_batch, k, h, w = X.shape
        oh, ow = delta.shape[2:]
        rh, rw = h / oh, w / ow
        ind = np.arange(s_batch * k * oh * ow) * rh * rw + self.__ind.flatten()
        return self.__backward(delta, ind, s_batch, k, h, w, oh, ow) * act.derivate(X)


    def __max(self, X, s_batch, k, oh, ow):
        patch = self.__im2patch(X, s_batch, k, oh, ow)
        return map(lambda _f: _f(patch, axis = 3).reshape(s_batch, k, oh, ow), [np.max, np.argmax])


    def __im2patch(self, X, s_batch, k, oh, ow):
        patch = np.zeros((s_batch, oh * ow, k, self.kh, self.kw))
        for j in range(oh):
            for i in range(ow):
                _j, _i = j * self.s, i * self.s
                patch[:, j * ow + i, :, :, :] = X[:, :, _j:_j+self.kh, _i:_i+self.kw]
        return patch.swapaxes(1, 2).reshape(s_batch, k, oh * ow, -1)


    def __backward(self, delta, ind, s_batch, k, h, w, oh, ow):
        _delta = np.zeros(s_batch * k * h * w)
        _delta[ind] = delta.flatten()
        return _delta.reshape(s_batch, k, oh, ow, self.kh, self.kw).swapaxes(3, 4).reshape(s_batch, k, h, w)
