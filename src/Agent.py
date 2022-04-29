# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 17:02:56 2021

@author: simon

Agent class that interacts with the environment
"""

import torch
import numpy as np
import random 

class Agent:
    def __init__(self,model):#seed
#         self.trajectory = torch.tensor([])
        self.model = model
        #self.set_seed(seed)
        self.X = None
        self.y = None
               

    def set_seed(self,new_seed):
        self.seed = new_seed
        # set seeds
        torch.manual_seed(new_seed)
        torch.cuda.manual_seed(new_seed)
        np.random.seed(new_seed)
        random.seed(new_seed)
        
    def create_init_state(self,batch,n,device=torch.device("cuda")):
        # creates the initial state vectors 

        
        states = torch.tensor([],device=device)
        for s in batch:
            kernel,param,loss = s
            
            kernels = torch.stack([torch.cat((kernel['mean_module.constant'],kernel['covar_module.raw_outputscale'].view(1),kernel['covar_module.base_kernel.raw_lengthscale'].flatten()))]).to(device=device)
            kernels = torch.cat((kernels[0].repeat(n-1,1),kernels)) # same padding, so the size is the same as all other state components        
        
            state = torch.cat((kernels,param,loss),1)
            states = torch.cat((states,state.unsqueeze(0)))
            
        # temporary variables to store dimensions of kernels and params
        self.kernel_size = kernels.shape[1]
        self.param_size = param.shape[1]
        
        return states
    
    def rollout(self,env,t,seeds=None,n=10,device=torch.device("cuda")):
        # Perform rollout with length t, evironment env, # of initial points n for seeds=seeds (batch size is given by number of seeds)
             
        # Get initial states from environment
        batch = []
        indices = torch.tensor([]).long()
        if type(seeds)!=list and type(seeds)!=torch.Tensor:
            seeds= [seeds]
        for seed in seeds:
            kernel,param,losses,initial_idxs = env.initGP(seed=seed,n=n)
            indices = torch.cat((indices,torch.tensor(initial_idxs).unsqueeze(0)),0)
            batch.append((kernel,param,losses))
            
        action_hist = torch.tensor([]).long()
        state_hist = {}
        kernel_hist = {}
        
        # add initial parameters to action_hist
        action_hist = torch.cat((action_hist,indices)).to(device=device)

        # create initial state vector
        states =  self.create_init_state(batch,n,device=device)

        # Main loop
        for step in range(t):
            print(f"\r{step}",end='')
            with torch.no_grad():
                # Forward through network
                pred = self.model(states.float())
                
            # pred to grid, make sure not to have same action again
            pred_idx,pred_param = env.pred_to_grid(pred,action_hist)
            
            action_hist = torch.cat((action_hist,pred_idx.unsqueeze(1)),axis=1)
            
            # transition to next state
            state_hist[step] = states
            states,kernel_state_dicts = env.transition(states,pred_param,action_hist,seeds,device=device)
            kernel_hist[step] = kernel_state_dicts
            
        # rollout finished, return states and actions
        return action_hist,state_hist,kernel_hist