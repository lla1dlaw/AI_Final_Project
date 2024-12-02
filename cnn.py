#cnn.py
#Mario

import torch.nn as nn

class CNN(nn.Module):
  def __init(self):
    super().__init__()
                        #(in, out, kernel)
    self.conv1 = nn.Conv2d(3,6,3)
    self.pool1 == nn.MaxPool2d(2,2)
    #if not resized v
    # m = nn.MaxPool2d((2, 2), stride=(2, 1))
    self.conv2 = nnConv2d(6,12,3)
    self.pool2 == nn.MaxPool2d(2,2)
    
    self.conv3 = nnConv2d(12,24,3)
    self.pool3 == nn.MaxPool2d(2,2)
    #images 433x288 maybe resize
  #TODO: determine if images will be resized before continuing 
    self.fc1 = nn.Linear(x*i*j,y)
    self.fc2 = nn.Linear(y,z)
    self.fc3 = nn.Linear(z,a)
  
  def forward(self,x):
    x = self.pool1(F.relu(self.conv1(x)))
    x = self.pool2(F.relu(self.conv2(x)))
    x = self.pool3(F.relu(self.conv3(x)))

    x = torch.flatten(x,1)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

  
