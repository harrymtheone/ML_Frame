import ML_Frame
from ML_Frame import layers
import numpy as np

data = np.zeros(1)


class Model:
    def __init__(self):
        self.fc1 = layers.Linear(1, 10)
        self.fc2 = layers.Linear(10, 1)
        self.relu = layers.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(x)
        return out


if __name__ == '__main__':
    pass



