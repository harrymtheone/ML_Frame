import numpy as np

from tensor import Tensor


class Module:
    def forward(self, x):
        pass

    def backward(self, grad_tensor):
        pass

    def __call__(self, x):
        self.forward(x)


class Conv2D(Module):
    def __init__(self):
        print('hhhh')

    def forward(self, x):
        pass

    def backward(self, grad):
        pass


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.input_features = in_features
        self.output_features = out_features
        self.bias = bias

        self.W = np.random.uniform(-0.1, 0.1, (out_features, in_features))
        if bias:
            self.b = np.zeros((out_features, 1))

    def forward(self, x):
        out = np.dot(self.W, x.data)
        if self.bias:
            out += self.b
        return Tensor(out, self, x)

    def backward(self, grad_tensor):
        # dx = np.dot(self.input_diff, self.W.T)
        # self.dW = self.lr * np.dot(self.dx.T, output_data)
        # self.db = self.lr * np.sum(dout, axis=0)
        #
        # dx = dx.reshape(self.output_features, self.input_features)  # 还原输入数据的形状（对应张量）
        # return dx
        pass


class Adder:
    def forward(self, x1, x2):
        out = x1.data + x2.data
        return Tensor(out, self, [x1, x2])

    def backward(self, grad_tensor):
        pass


class ReLU(Module):
    def forward(self, x):
        out = np.heaviside(x.data, 0) * x.data
        return Tensor(out, self, x)

    def backward(self, grad_tensor):
        pass
