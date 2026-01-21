"""
Extracts the Encoder and the Decoder from the fully convolutional mesh autoencoder.
"""

from typing import Any
import torch.nn as nn


class Encoder:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out = self.model.forward_till_layer_n(x, len(self.model.channel_lst) // 2)
        return out
        
# class Encoder(nn.Module):
#     def __init__(self, model):
#         super(Encoder, self).__init__()
#         self.model = model

#     def forward(self, x):
#         out = self.model.forward_till_layer_n(x, len(self.model.channel_lst) // 2)
#         return out

class Decoder:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out = self.model.forward_from_layer_n(x, len(self.model.channel_lst) // 2)
        return out

# class Decoder(nn.Module):
#     def __init__(self, model):
#         super(Decoder, self).__init__()
#         self.model = model

#     def forward(self, x):
#         out = self.model.forward_from_layer_n(x, len(self.model.channel_lst) // 2)
#         return out
