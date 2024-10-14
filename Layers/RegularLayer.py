import torch
import torch.nn.functional as func

from Layers.Layer import Layer


class RegularLayer(Layer):
    def __init__(self, num_inputs, num_outputs):
        # gaussian
        self.weights = torch.randn(num_inputs, num_outputs)
        self.bias = torch.randn(num_outputs)
        self.outputs = None
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = torch.sigmoid(torch.mm(inputs, self.weights) + self.bias)
        return self.outputs

    def backward(self, grad_output, learning_rate=0.01):
        sigmoid_derivative = self.outputs * (1 - self.outputs)

        errors = torch.mm(sigmoid_derivative.T, sigmoid_derivative)  # This depends on the layer's output

        grad_weights = torch.mm(self.inputs.T, errors)
        grad_bias = torch.sum(errors, dim=0)

        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return torch.mm(errors, self.weights.T)



