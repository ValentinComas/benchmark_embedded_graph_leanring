import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np

class GatedGraphNetwork(nn.Module):
    def __init__(self, input, hidden, output, n_layers, bi=False):
        super(GatedGraphNetwork, self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.tanh = nn.Tanh()
        self.rec_unit = nn.GRU(input_size=input, hidden_size=hidden, num_layers=n_layers, batch_first=True, dropout=0.3, bidirectional=bi)
        self.fc = nn.Linear(hidden, output)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.rec_unit(x, h)
        out = out[:,-1]
        out = self.tanh(out)
        out = self.fc(out)
        out = self.relu(out)
        return out, h
    
    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden).zero_().to(device),
                    weight.new(self.n_layers, batch_size, self.hidden).zero_().to(device))
        return hidden

class LstmNetwork(nn.Module):
    def __init__(self, input, hidden, output, n_layers, bi=False):
        super(LstmNetwork, self).__init__()
        self.rnn_type = rnn_type
        self.hidden = hidden
        self.n_layers = n_layers
        self.tanh = nn.Tanh()
        self.rec_unit = nn.LSTM(input_size=input, hidden_size=hidden, num_layers=n_layers, batch_first=True, dropout=0.3, bidirectional=bi)
        self.fc = nn.Linear(hidden, output)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.rec_unit(x, h)
        out = out[:,-1]
        out = self.tanh(out)
        out = self.fc(out)
        out = self.relu(out)
        return out, h
    
    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden).zero_().to(device),
                    weight.new(self.n_layers, batch_size, self.hidden).zero_().to(device))

# class ConvolutionalGraphNetwork(nn.Module):
#     def __init__(self, output):
#         super(ConvolutionalGraphNetwork, self).__init__()
#         self.conv1 = nn.Conv2d(2, 32, kernel_size=5)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
#         self.lin = nn.Linear(64 * 5 * 5, 32)
#         self.lin2 = nn.Linear(32, output)
#         self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.drop = nn.Dropout2d(0.25)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
        
        
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.max_pool(out)
#         out = self.conv2(out)
#         out = self.drop(out)
#         out = self.max_pool(out)
        
#         out = out.view(out.size(0), -1)

#         out = self.lin(out)
#         out = self.tanh(out)
#         out = self.lin2(out)
#         out = self.relu(out)
#         return out