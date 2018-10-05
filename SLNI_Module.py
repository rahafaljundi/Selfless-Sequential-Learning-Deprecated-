
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
import torch.utils.data as data
import torchvision
from torchvision import datasets, models, transforms
import scipy.stats as stats
import sys


import pdb

#

class SLNI_Module(nn.Module):
    """
    A wrapper over an existing architicture to apply the SLNI loss
    """
    def __init__(self, module, sparcified_layers):
        super(SLNI_Module, self).__init__()
        self.module = module
        self.sparcified_layers=sparcified_layers
        self.neuron_omega=False
        self.scale=0
        self.squash='exp'#squashing function exp or sigmoid


    def forward(self, x):
        SLNI_loss =0
        sub_index=0
        
        for name, module in self.module._modules.items():
        
                for namex, modulex in module._modules.items():
                    
                    x = modulex(x)
                    if namex in self.sparcified_layers[sub_index]:
                        
                        if hasattr(modulex, 'neurons_importance'):
                            #After the first task, the importance of the neurons would have been computed
                            
                            
                            neurons_importance=modulex.neurons_importance
                            if self.scale==0:
                                #SNI no gaussian weighting
                                SLNI_loss +=SNI_loss_neuron_importance(x,neurons_importance,self.min,self.squash) 
                            else:
                                SLNI_loss += SLNI_loss_neuron_importance(x,neurons_importance,x.size(1)/self.scale,self.squash)
                        else:
                            if self.scale==0:
                                SLNI_loss +=SNI_loss(x)#SNI NO GAUSSIAN WEIGHTING
                            else:
                                SLNI_loss += SLNI_loss(x,x.size(1)/self.scale)
                
                #for reshaping the fully connected layers
                if sub_index==0:
                    #this is a hack for architictures like Alexnet and VGG, a reshaping before the FC
                    try:
                        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
                    except:
                        pass
                sub_index+=1
        #pdb.set_trace()        
        return x,SLNI_loss

    
    
    
def SNI_loss(A):  
  
    """
    No Gaussian Scaling
    """

    
    cov=(1/A.size(0))*torch.mm(torch.transpose(A,0,1),A)
    cov_norm=cov.norm(1)
    diag=torch.eye(cov.size(0)).cuda()
    diag=Variable(diag, requires_grad=False)
    cov_diag=cov*diag
    cov_diag_norm=cov_diag.norm(1)
    SNI_loss=(cov_norm-cov_diag_norm)

    return SNI_loss

def SNI_loss_neuron_importance(A,neuron_omega_val,squash='exp'):
    """
       No Gaussian Scaling, with neuron importance after the first task
    """
    
    sigmoid=torch.nn.Sigmoid()
    if squash=='exp':
        y=torch.exp(-neuron_omega_val)#USED IN THE PAPER
    else:
        
        y=1- sigmoid(neuron_omega_val)
        y=(y-y.min())/(y.max()-y.min())
                                  

    y=Variable(y.data, requires_grad=False)

    Az=torch.mul(y,A)
    cov=(1/Az.size(0))*torch.mm(torch.transpose(Az,0,1),Az)
    cov_norm=cov.norm(1)
    diag=torch.eye(cov.size(0)).cuda()
    diag=Variable(diag, requires_grad=False)
    cov_diag=cov*diag
    cov_diag_norm=cov_diag.norm(1)
    SNI_loss=(cov_norm-cov_diag_norm)
  
    return SNI_loss

def SLNI_loss(A,scale=32):

    cov=(1/A.size(0))*torch.mm(torch.transpose(A,0,1),A)
    normal_weights=np.fromfunction(lambda i, j: stats.norm.pdf((i-j), loc=0, scale=scale)/stats.norm.pdf(0, loc=0, scale=scale), cov.size(), dtype=int)
    normal_weights=torch.Tensor(normal_weights).cuda()
    cov=cov*normal_weights
    cov_norm=cov.norm(1)
    diag=torch.eye(cov.size(0)).cuda()
    diag=Variable(diag, requires_grad=False)
    cov_diag=cov*diag

    cov_diag_norm=cov_diag.norm(1)
    SLNI_loss=(cov_norm-cov_diag_norm)
    
    return SLNI_loss


def SLNI_loss_neuron_importance(A,neuron_omega_val,scale,squash='exp'):
  
    sigmoid=torch.nn.Sigmoid()
    if squash=='exp':
        y=torch.exp(-neuron_omega_val)#USED IN THE MAIN PAPER
    else:
        
        y=1- sigmoid(neuron_omega_val)
        y=(y-y.min())/(y.max()-y.min())
                                  

    y=Variable(y.data, requires_grad=False)

    Az=torch.mul(y,A)
    cov=(1/Az.size(0))*torch.mm(torch.transpose(Az,0,1),Az)

    normal_weights=np.fromfunction(lambda i, j: stats.norm.pdf((i-j), loc=0, scale=scale)/stats.norm.pdf(0, loc=0, scale=scale), cov.size(), dtype=int)
    normal_weights=torch.Tensor(normal_weights).cuda()
    cov=cov*normal_weights
    cov_norm=cov.norm(1)
    diag=torch.eye(cov.size(0)).cuda()
    diag=Variable(diag, requires_grad=False)
    cov_diag=cov*diag
    cov_diag_norm=cov_diag.norm(1)
    SLNI_loss=(cov_norm-cov_diag_norm)

    return SLNI_loss
