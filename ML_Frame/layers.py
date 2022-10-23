import numpy as np

class Conv2D:
    def __init__(self):
        print('hhhh')

class Fully_connected_layer:
    def __init__(self, input_size, output_size, activator,lr):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.b = np.zeros((output_size,1))
        self.lr = lr

    def forward(self, input_data):
        self.input_data = input_data
        # y = sigmoid(W*x+b)
        self.output_data = self.activator(np.dot(self.W, self.input_data) + self.b)

    def backward(self, output_data, input_diff):
        dx = np.dot(self.input_diff, self.W.T)
        self.dW = self.lr * np.dot(self.dx.T, output_data)
        self.db = self.lr * np.sum(dout, axis=0)
        
        dx = dx.reshape(self.output_size, self.input_size)  # 还原输入数据的形状（对应张量）
        return dx






