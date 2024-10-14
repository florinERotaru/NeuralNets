from math import floor

import torch
import numpy as np
from torch.autograd import backward
from torchvision import datasets

from Layers.RegularLayer import RegularLayer
from Network import Network

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=None
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=None
)


# we want to obatin n pairs
# of ((b_size, 28*28), labels)
def prepare_batches(dataset, batch_size):
    batches = []
    n = floor(len(dataset) / batch_size)
    for i in range(n):
        batch_images = []
        batch_labels = []
        for j in range(batch_size * i, batch_size * (i + 1)):
            if j >= len(dataset):
                continue
            batch_image = np.array(dataset[j][0])

            #convert to list since appending to a list will be faster
            batch_image = batch_image.flatten().tolist()

            batch_images.append(batch_image)
            batch_labels.append(dataset[j][1])
        #add batch pair to batches
        batches.append(
            (
                torch.tensor(batch_images),
                torch.tensor(batch_labels)
            ),
        )
    return batches

def standardize(dataset):
    all_images = np.array([np.array(data[0]) for data in dataset])

    mean = np.mean(all_images)
    std = np.std(all_images)

    standardized_dataset = []
    for image, label in dataset:
        standardized_image = (np.array(image) - mean) / std
        standardized_dataset.append((standardized_image, label))

    return standardized_dataset

train_dataset_standardized = standardize(train_dataset)
test_dataset_standardized = standardize(test_dataset)

train_batches = prepare_batches(train_dataset_standardized, 256)
test_batches = prepare_batches(test_dataset_standardized, 256)

n = Network()
n.add_hidden_layer(RegularLayer(784, 100))
n.add_output_layer(RegularLayer(100, 10))

# output = n.backward(train_batches[0][0], train_batches[0][1])
output = n.process_batch(train_batches[0][0], train_batches[0][1])
# print(output.shape)
n.backward(output, train_batches[0][1])

