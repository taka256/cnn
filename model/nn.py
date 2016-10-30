import numpy as np

class NN(object):

    def __init__(self, n_input, n_hidden, n_output, input_act, hidden_act, output_act):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.hidden_weight = np.random.randn(n_hidden, n_input + 1) * 0.01
        self.output_weight = np.random.randn(n_output, n_hidden + 1) * 0.01
        self.input_act = input_act
        self.hidden_act = hidden_act
        self.output_act = output_act


    def forward(self, x):
        z = self.__forward(x, self.hidden_weight, self.hidden_act)
        y = self.__forward(z, self.output_weight, self.output_act)
        return z, y


    def backward(self, t, y, z, u, error):
        output_delta = y - t # error.derivated_delta(t, y) * self.output_act.derivate(y)
        hidden_delta = self.__delta(self.output_weight, output_delta, z, self.hidden_act)
        input_delta = self.__delta(self.hidden_weight, hidden_delta, u, self.input_act)
        return output_delta, hidden_delta, input_delta


    def update_weight(self, output_delta, hidden_delta, z, u, epsilon, lam):
        s_batch = z.shape[1]
        reg_term = np.hstack((np.zeros((self.n_output, 1)), self.output_weight[:, 1:]))
        self.output_weight -= epsilon * (self.__grad(z, output_delta, s_batch) + lam * reg_term)
        self.hidden_weight -= epsilon * self.__grad(u, hidden_delta, s_batch)


    def __forward(self, x, weight, act):
        return act.activate(weight.dot(np.vstack((np.ones((x.shape[1])), x))))


    def __delta(self, weight, delta, x, act):
        return weight[:, 1:].T.dot(delta) * act.derivate(x)


    def __grad(self, x, delta, s_batch):
        return delta.dot(np.vstack((np.ones((1, x.shape[1])), x)).T) / s_batch
