import torch
import torch.nn.functional as func

class Network:

    def __init__(self):
        self.output_layer = None
        self.hidden_layers = []

    def add_hidden_layer(self, hidden_layer):
        self.hidden_layers.append(hidden_layer)

    def add_output_layer(self, output_layer):
        self.output_layer = output_layer

    def process_batch(self, X, y):
        outputs = None
        for hidden_layer in self.hidden_layers:
            outputs = hidden_layer.forward(X)
        outputs = self.output_layer.forward(outputs)

        return outputs

    def backward(self, outputs, y, learning_rate=0.01):
        errors = outputs.clone()
        for i in range(len(y)):
            errors[i][y[i]] -= 1
        print(self.l)
        print(errors.shape)
        # weights_grad =


