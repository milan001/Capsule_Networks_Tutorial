#!/usr/bin/env python
# coding: utf-8

# In[19]:


import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


# In[14]:


import numpy as np
A=np.ones(shape=(2,2))
print(A[:])
B=np.matmul(A[None,:,:], A[:,:,None])
print(np.size(B,(0,1)))


# In[31]:


class capsLayer(nn.Module):
    def __init__(self, input_channels, output_channels, num_iterations=20, num_capsules=1,
                 primary=False, kernel_size=None, stride=None, num_route_nodes=-1, caps_dim=-1):
        super(capsLayer, self).__init__()
        self.primary=primary
        self.input_channels=input_channels
        self.output_channels=output_channels
        if primary:
            self.caps_dim=caps_dim
            self.kernel_size=kernel_size
            self.stride=stride
        else:
            self.num_capsules=num_capsules
            self.num_iterations=num_iterations
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, input_channels, output_channels))
        
    def squash(self, x, dim):
        squared_norm=(x**2).sum(dim=dim, keepdim=True)
        return (squared_norm/(1+squared_norm))*x/torch.sqrt(squared_norm)
        
    def forward(self, x):
        if not self.primary:
            U=torch.mm(x[:, None, :, None, :], self.route_weights[None, :, :, :, :])
            B=Variable(torch.zeros(U.size()))
            for i in range(self.num_iterations):
                C=F.softmax(B, dim=1)
                V=(C*U).sum(dim=2, keepdim=True)
                V=V.squash(dim=-1)
                B=B+U*V
            V=V.view(B.size(0), B.size(1), B.size(-1))
        else:
            V=[F.conv2d(x, input_channels, output_channels, kernel_size=self.kernel_size,
                        stride=self.stride, padding=0).view(x.size(0), -1, 1) for _ in range(self.caps_dim)]
            V.cat(V, dim=-1)
            V.squash(V,  dim=-1)
        return V


# In[36]:



class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = capsLayer(caps_dim=8, primary=True, input_channels=256, output_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = capsLayer(num_capsules=10, num_route_nodes=32 * 6 * 6, input_channels=8,
                                           output_channels=16)

        self.decoder = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            if torch.cuda.is_available():
                y = Variable(torch.eye(10)).cuda().index_select(dim=0, index=max_length_indices)
            else:
                y = Variable(torch.eye(10)).index_select(dim=0, index=max_length_indices)
        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        return classes, reconstructions


if __name__ == "__main__":
    model = CapsuleNet()
print(model)


# In[ ]:




