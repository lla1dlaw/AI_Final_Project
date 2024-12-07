# Filename: cnn.py
# Author: Mario Slaybe
# Created: 12-02-2024
# Description: A convolutional network to predict genre of song based on spectrographic models of sound data
# note: #images 432x288




# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CNN(nn.Module):
#   def __init(self, layers: int, hidden_size: int, out_size: int):
#     super().__init__()
    
#     self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#     self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#     self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#     self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#     self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#     self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

#     self.linear = [nn.Linear(109 * 73 * 128, hidden_size)]
#     self.linear.extend([nn.Linea# Filename: cnn.py
# Author: Mario Slaybe
# Created: 12-02-2024
# Description: A convolutional network to predict genre of song based on spectrographic models of sound data
# note: #images 432x288

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, out_size: int):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 54 * 36, 512)
        self.fc2 = nn.Linear(512, out_size)

        # TODO: determine if images will be resized before continuing



    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


r(hidden_size, hidden_size) for i in range(layers)])
#     self.linear.append(nn.Linear(hidden_size, out_size))
#     self.linear = nn.ModuleList(self.linear)
#     self.activation = nn.ReLU()

#     #TODO: determine if images will be resized before continuing 

#     # alternative network
#     #(in, out, kernel)
#     # self.conv1 = nn.Conv2d(3,6,3)
#     # self.pool1 == nn.MaxPool2d(2,2)
#     # #if not resized v
#     # # m = nn.MaxPool2d((2, 2), stride=(2, 1))
#     # self.conv2 = nn.Conv2d(6,12,3)
#     # self.pool2 == nn.MaxPool2d(2,2)
    
#     # self.conv3 = nn.Conv2d(12,24,3)
#     # self.pool3 == nn.MaxPool2d(2,2)

#     # self.fc1 = nn.Linear(x*i*j,y)
#     # self.fc2 = nn.Linear(y,z)
#     # self.fc3 = nn.Linear(z,a)
  
#   def forward(self,x):
#     x = self.pool1(F.relu(self.conv1(x)))
#     x = self.pool2(F.relu(self.conv2(x)))
#     x = self.pool3(F.relu(self.conv3(x)))

#     x = torch.flatten(x,1)
#     for layer in self.linear:
#             x = layer(x)
#             x = F.relu(x)
    
#     return x
  
#     # alternative forward pass
#     # x = F.relu(self.fc1(x))
#     # x = F.relu(self.fc2(x))
#     # x = self.fc3(x)

  
