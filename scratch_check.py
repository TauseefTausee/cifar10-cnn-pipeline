"""Scratch file: quick check that CIFAR-10 downloads and has the expected shapes.

Only used while exploring the dataset. prepare.py replaces it.
"""
from torchvision.datasets import CIFAR10

dataset = CIFAR10(root=".cache/cifar10", train=True, download=True)
print("images:", dataset.data.shape)
print("labels:", len(dataset.targets))
print("classes:", dataset.classes)
