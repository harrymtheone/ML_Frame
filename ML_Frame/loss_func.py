import numpy as np
from tensor import Tensor


class MSE:
    def __call__(self, prediction: Tensor, target: Tensor):
        loss = np.mean(np.square(prediction.data - target.data))
        return Tensor(loss, self, [prediction, target])

    @staticmethod
    def backward(loss):
        prediction = loss.parent_tensors[0].data
        target = loss.parent_tensors[1].data
        grad = 2 * np.mean(prediction - target)
        loss.parent_tensors[0].grad = grad
        loss.parent_tensors[0].backward()


