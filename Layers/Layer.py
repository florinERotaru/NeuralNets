from abc import abstractmethod

class Layer:

    @abstractmethod
    def __init__(self, num_inputs, num_outputs):
        pass
