# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 14:53:53 2022

@author: simon

Custom Dataset class to supply differently sized states (storing the history so far)
"""
import torch

class History_Increasing_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, _dataset):
        # _dataset needs to be a dict of tensor tuples {size: (torch.tensor([len x size x state_dim]),torch.tensor([len x action_size]))} -> as the first dimension of the state vector varies
        self.dataset = _dataset
        self.current_hist_length = 0

    def __getitem__(self, index):
        state = self.dataset[self.current_hist_length][0][index]
        action = self.dataset[self.current_hist_length][1][index]
        return state,action

    def __len__(self):
        return len(self.dataset[self.current_hist_length][0])

    def sethistlength(self,H):
        self.current_hist_length = H
        

        
class Non_History_Increasing_Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, _dataset):
      self.dataset = _dataset

    def __getitem__(self, index):
        state = self.dataset[self.current_hist_length][0][index]
        action = self.dataset[self.current_hist_length][1][index]
        return state,action

    def __len__(self):
        return len(self.dataset[self.current_hist_length][0])

    def sethistlength(self,H):
        self.current_hist_length = H