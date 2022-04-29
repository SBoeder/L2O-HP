# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:53:16 2021

@author: simon

Environment for HPO:
    - implements transitions
    - implements training of kernels on current dataset
"""

import torch
import numpy as np
import random
from Utility import fit_simple_custom_gp,load_GP,get_expert_actions
from sklearn.preprocessing import StandardScaler
import gc

class Environment:
    # Environment class for the agent to interact with
    def __init__(self,X,y):
        self.X = X
        self.y = y
        self.seed = None
        
        # Scaler for GP fitting
        self.sc = StandardScaler()
        self.scaledX = torch.tensor(self.sc.fit_transform(X.cpu()),device=X.device)
        
    def set_seed(self, seed):
        self.seed = seed
        # set seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(int(seed))
        random.seed(seed)
        
        self.rng = np.random.default_rng(int(seed))
        
    def get_number_of_actions(self):
        return min(100,self.X.shape[0]-10)
        
    def next_kernel(self,action,trajectory,seeds,device=torch.device("cuda")):    
        # train next GP kernel from action
        Xs = self.scaledX[trajectory]
        ys = self.y[trajectory]
        next_kernels = torch.tensor([],device=device)
        kernel_dicts = []
        for batch in range(Xs.shape[0]):
            # TODO can we somehow train multiple GPs in parallel? -> batch mode?
            try:
                new_kernel = fit_simple_custom_gp(Xs[batch], ys[batch],seeds[batch],device=torch.device("cpu")) # train GP on CPU               
                kernel_state_dict = new_kernel.state_dict()
                kernel_dicts.append(kernel_state_dict)
                kernels = torch.cat((kernel_state_dict['mean_module.constant'],kernel_state_dict['covar_module.raw_outputscale'].view(1),kernel_state_dict['covar_module.base_kernel.raw_lengthscale'].flatten())).to(device=device)
                next_kernels = torch.cat((next_kernels,kernels.unsqueeze(0)))
            except:
                kernel_dicts.append("CHOLESKY")
        return next_kernels,kernel_dicts
    
    def pred_to_grid(self,actions,hist,delta_next_action=True):
        # get the last params
        previous_params = self.X[hist[:,-1]]
        # Bring continous predictions of model to grid with nearest neighbor method
        next_params = previous_params + actions
        gridded_actions_indx = torch.cdist(next_params, self.X)
        # mask actions already taken
        gridded_actions_indx = gridded_actions_indx.scatter(1,hist,np.inf)
        # take argmin
        gridded_actions_indx = torch.argmin(gridded_actions_indx,axis=1)
        gridded_actions_params = self.X[gridded_actions_indx]
        return gridded_actions_indx,gridded_actions_params
    
    def initGP(self,seed=None, n=10,inital_idxs=None,device=torch.device("cpu")):
        # Function to sample initial params
        # draw random hyperparameter combination indices as initial points for the GP
        
        if self.seed == None and seed==None:
            print("No seed set, use function set_seed(seed) or hand seed as parameter to this function")
            return
        if seed != None:
            self.set_seed(seed)
            
        if inital_idxs != None:
            # Use these initial parameters
            idxs = inital_idxs
        else:
            randidx = np.arange(0,len(self.X)) 
            self.rng.shuffle(randidx)
            idxs = randidx[:n]

        # Fit initial GP
        model = fit_simple_custom_gp(self.scaledX[idxs],self.y[idxs],self.seed,device=device)
        
        losses = self.y[idxs]
        params = self.X[idxs]
        return model.state_dict(),params,losses,idxs
        
    def transition(self,states,actions,trajectory,seeds,device=torch.device("cuda")):
        # get the next kernel given the action
        new_kernels,kernel_state_dicts = self.next_kernel(actions, trajectory, seeds,device)
        
        for i,kernel in enumerate(kernel_state_dicts):
            # if we hit cholesky error at new point, replace kernel with last kernel
            if kernel == "CHOLESKY":
                new_kernels = torch.cat((new_kernels[:i],states[i,-1,:-(actions.shape[1] +1)].unsqueeze(0),new_kernels[i:]))
        next_set_element = torch.cat((new_kernels,actions,self.y[trajectory[:,-1]]),axis=1)
        next_state = torch.cat((states,next_set_element.unsqueeze(1)),axis=1)
        return next_state, kernel_state_dicts
        
        
    def expert_actions(self,trajectory,states,kernel_state_dicts,seeds,device=torch.device("cuda")):
        # returns the actions that the teacher acquisition functions would have taken in every step
        if type(seeds) != list and type(seeds)!= torch.Tensor:
            seeds = [seeds]
        T = self.get_number_of_actions()
        actions_hist = {k:torch.tensor([]) for k in range(T)}
        print("Getting actions...",flush=True)
        for t in range(T):
            print(f"\r{t}",end='')
            # print("Reserved: {}".format(torch.cuda.memory_reserved(0) / 1e9))
            # print("Allocated: {}".format(torch.cuda.memory_allocated(0)/ 1e9))
            # print()
            torch.cuda.empty_cache()
            
             # check if the kernel state dict of this step was a cholesky error, if yes, skip this step
            if "CHOLESKY" in kernel_state_dicts[t]:
                continue
            for i,seed in enumerate(seeds):
                GP_model = load_GP(self.scaledX,self.y,kernel_state_dicts[t][i],seed)
                best_f = self.y[trajectory[i,:t+10]].max().item()
                actions = get_expert_actions(GP_model,self.scaledX.unsqueeze(1),best_f,trajectory[i,:t+10].cpu(),device=device)
                if actions.shape[0] < 4:
                    # Error while fetching action, reset to no actions on this timestep
                    actions_hist[t] = torch.tensor([])
                    break
                difference_in_X = (self.X[actions] - self.X[trajectory[i][t+9]]).detach().cpu()# +9 here, as this is inclusive
                actions_hist[t] = torch.cat((actions_hist[t],difference_in_X.unsqueeze(0)))
                del GP_model,difference_in_X,actions
            gc.collect()
                
        return actions_hist
                