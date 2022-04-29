# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:59:57 2021

@author: simon

All model variants of the GPS for Black-box HPO 

DEFAULT variant is "Learner" --> combines all parts of the state and takes them as input concatted
"""

import torch
import torch.nn as nn
from set_transformer.modules import SAB, PMA, ISAB

class Learner(nn.Module):
    def __init__(self,state_size,action_size,hidden_size=64,transformer_hidden_size=128,transformer_out_features=64,heads=4,n_hidden_layers=1):
        super(Learner,self).__init__()
      
        
        self.settransformer = nn.Sequential(
            SAB(dim_in=state_size,dim_out=transformer_hidden_size,num_heads=heads),
            SAB(dim_in=transformer_hidden_size,dim_out=transformer_hidden_size,num_heads=heads),
            PMA(dim=transformer_hidden_size, num_heads=heads, num_seeds=1),
            SAB(dim_in=transformer_hidden_size,dim_out=transformer_hidden_size,num_heads=heads),
            SAB(dim_in=transformer_hidden_size,dim_out=transformer_hidden_size,num_heads=heads),
            nn.Linear(in_features=transformer_hidden_size, out_features=transformer_out_features)
        )
        
        # FF network list
        networklist = [nn.Linear(transformer_out_features, hidden_size),nn.Softplus()] # input layer
        for layer in range(n_hidden_layers): # hidden layers, amount based on hyperparameters
            networklist.append(nn.Linear(hidden_size,hidden_size))
            networklist.append(nn.Softplus())
        networklist.append(nn.Linear(hidden_size, action_size)) # output layer
        
       
        # Build FF network from list
        self.network = nn.Sequential(*networklist)

        
    def forward(self,state):
        
        x = self.settransformer(state).squeeze(1)     
        
        x = self.network(x)
        
        return x        
    
    
class Learner_indep(nn.Module):
    # old version of the learner that learns a set transformer for every state component individually.
    # this loses the attention information between the state components completely
    
    def __init__(self,paramsize,kernelsize,action_size,hidden_size=64,transformer_hidden_size=64,transformer_out_features=64,heads=4):
        super(Learner_indep,self).__init__()
        
        # Set transformer -> Permutation equivariance
        self.settransformer_params = nn.Sequential(
            SAB(dim_in=paramsize,dim_out=transformer_hidden_size,num_heads=heads),
            SAB(dim_in=transformer_hidden_size,dim_out=transformer_hidden_size,num_heads=heads),
            PMA(dim=transformer_hidden_size, num_heads=heads, num_seeds=1),
            #SAB(dim_in=transformer_hidden_size,dim_out=transformer_hidden_size,num_heads=heads),
            nn.Linear(in_features=transformer_hidden_size, out_features=transformer_out_features)
        )

        self.settransformer_kernels = nn.Sequential(
            SAB(dim_in=kernelsize,dim_out=transformer_hidden_size,num_heads=heads),
            SAB(dim_in=transformer_hidden_size,dim_out=transformer_hidden_size,num_heads=heads),
            PMA(dim=transformer_hidden_size, num_heads=heads, num_seeds=1),
            #SAB(dim_in=transformer_hidden_size,dim_out=transformer_hidden_size,num_heads=heads),
            nn.Linear(in_features=transformer_hidden_size, out_features=transformer_out_features)
        )
        self.settransformer_loss = nn.Sequential(
            SAB(dim_in=1,dim_out=transformer_hidden_size,num_heads=heads),
            SAB(dim_in=transformer_hidden_size,dim_out=transformer_hidden_size,num_heads=heads),
            PMA(dim=transformer_hidden_size, num_heads=heads, num_seeds=1),
            #SAB(dim_in=transformer_hidden_size,dim_out=transformer_hidden_size,num_heads=heads),
            nn.Linear(in_features=transformer_hidden_size, out_features=transformer_out_features)
        )


        # Actual FF network
        self.network = nn.Sequential(
            nn.Linear(transformer_out_features*3, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, action_size),
        )
        
    def forward(self,x_kernels,x_params,x_losses):
        x_kernels = self.settransformer_kernels(x_kernels)
        x_params = self.settransformer_params(x_params)
        x_losses = self.settransformer_loss(x_losses)
                
        x_kernels = x_kernels.view(x_kernels.size(0),-1)
        x_params = x_params.view(x_params.size(0),-1)
        x_losses = x_losses.view(x_losses.size(0),-1)
       
        
        x = torch.cat((x_kernels, x_params, x_losses), dim=1)

        x = self.network(x)
        return x        