import torch.nn as nn
import torch
# import torch.nn.functional as F


class Feedforward(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.output_size, bias=True)
        # torch.nn.init.normal_(self.l1.weight, mean=0, std=1)
        # torch.nn.init.xavier_uniform_(self.l1.weight, gain=1.0)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x = x.view(x.size(0), -1)  # convert tensor (nb_atom, 1, nb_neighbors,
        # nb_prop) --> (nv_atom,nb_neighbors*nb_prop)
        # convert tensor (nb_neighbors, nb_prop) --> (nb_neighbors*nb_prop)
        x = x.view(-1)
        output = self.l1(x)
        output = self.softmax(output)
        return output
