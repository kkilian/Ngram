import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
class OneHotEncoder:
    def __init__(self, keys):
        self.keys = keys
        self.one_hot_encoded = {}
        self.encode()

    def encode(self):
        for key in self.keys:
            one_hot_tensor = torch.zeros(len(self.keys), dtype=torch.float32)
            one_hot_tensor[self.keys.index(key)] = 1.0
            self.one_hot_encoded[key] = one_hot_tensor

    def get_encoded(self):
        return self.one_hot_encoded

    def set_values_to_integers(self, dictionary):
        for i, key in enumerate(dictionary.keys()):
            dictionary[key] = i
        return dictionary
