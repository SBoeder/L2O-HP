# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 12:03:35 2021

@author: simon

Data generation script for "GPS for Blackbox functions"
"""

import torch
# import gpytorch
import pickle as pkl
import os
import numpy as np
import re

from sklearn.preprocessing import StandardScaler
import sys


from HPO_B import hpob_handler
import time

import argparse
import gc


# Ignore filters
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
    
# Track runtime
start_time = time.time() 

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d',"--dataset")
parser.add_argument("-k","--keys",nargs="+")
parser.add_argument("-o","--done",nargs="+")
parser.add_argument("-n","--only",nargs="+") # specific datasets of this hpo_b key
args = parser.parse_args()
print("arguments read",flush=True)
if args.done == None:
    args.done = []
# use a GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") # comment this to use GPU
dtype = torch.float
 
# Load data
rootdir = ".."

# hpo_AA
if args.dataset == "hpo_data":
    data_file = os.path.join(rootdir, "AA.pkl")
    with open(data_file, "rb") as f:
        hpo_data = pkl.load(f)
    
# Load hpo_benchmark
if args.dataset == "hpo_b":
    hpob_hdlr = hpob_handler.HPOBHandler(root_dir=os.path.join(rootdir,"hpob-data/"), mode="v2")
    ids = hpob_hdlr.get_search_spaces()

print("data loaded",flush=True)

# Load and build GP model
from botorch.models import SingleTaskGP
from botorch.acquisition.analytic import UpperConfidenceBound, ProbabilityOfImprovement
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
import gpytorch.settings
from botorch.optim.fit import fit_gpytorch_torch

# Define fit function
def fit_simple_custom_gp(Xs, Ys,seed):
    torch.manual_seed(seed)
    model = SingleTaskGP(Xs, Ys).to(device)
    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(device)
    # kwargs = {"track_iterations":False}
    # fit_gpytorch_model(mll,optimizer=fit_gpytorch_torch,**kwargs)
    mll.train()
    fit_gpytorch_torch(mll,track_iterations =False)
    mll.eval()
    return model

# Define EI evaluation function, which in this case returns real response from metadata set
def evaluation_EI(model,data,best_f,H=None):
    model.eval()
    model.likelihood.eval()
    EI = ExpectedImprovement(model,best_f=best_f)
    EI_range = EI(data)
    
    # generate candidate
    if H is not None:
        EI_range[H]= -np.inf
        return torch.argmax(EI_range)
    else:
        return EI_range
    
# Define UCB evaluation function, which in this case returns real response from metadata set
def evaluation_UCB(model,data,b=0.2,H=None):
    model.eval()
    model.likelihood.eval()
    UCB = UpperConfidenceBound(model, beta=b)
    UCB_range = UCB(data)
    
    # generate candidate
    if H is not None:
        UCB_range[H]= -np.inf
        return torch.argmax(UCB_range)
    else:
        return UCB_range
    
def evaluation_MES(model,data,H=None):
    model.eval()
    model.likelihood.eval()
    MES = qMaxValueEntropy(model,data.squeeze(1))
    mes = MES(data)
    
    # generate candidate
    if H is not None:
        mes[H]= -np.inf
        return torch.argmax(mes)
    else:
        return mes
    
# probability of improvement
def evaluation_POI(model,data,best_f,H=None):
    model.eval()
    model.likelihood.eval()
    POI = ProbabilityOfImprovement(model,best_f=best_f)
    POI_range = POI(data)
    
    # generate candidate
    if H is not None:
        POI_range[H]= -np.inf
        return torch.argmax(POI_range)
    else:
        return POI_range
    
# Function to sample initial params
def initGP(X,y, rng,seed, n=5):
    
    # draw random hyperparameter combination indices as initial points for the GP
    randidx = np.arange(0,len(X)) 
    rng.shuffle(randidx)
    idxs = randidx[:n]
    
    # Fit initial GP
    model = fit_simple_custom_gp(X[idxs],y[idxs],seed)
    return model,idxs

# Function to generate sequences for given data and seed for each acquisition function
def GenerateSequence(X,y,T,seed = 42):
            
    # Standardize input data
    sc_inp = StandardScaler()
    sc_outp = StandardScaler().fit(y)
    X = torch.tensor(sc_inp.fit_transform(X),device=device)
    # y = torch.tensor(sc_outp.transform(y),device=device)
    
    # Loop over all acquisition functions (UCB, EI, ES, PoI)
    acqfuns = [evaluation_UCB,evaluation_EI,evaluation_MES,evaluation_POI]
    improvement_hist ={}
    loss_hist = {}
    param_hist = {}
    kernel_hist = {}
    n=10
    for acqfun in acqfuns:
        print("Start acqf {}".format(acqfun.__name__))
        # init the rng
        rng = np.random.default_rng(seed)
        
        # Init GP model
        m,H = initGP(X,y,rng,seed,n=n)
        # losses = [max(sc_outp.inverse_transform(y[H]))]
        losses = [max(y[H])]
        # reallosses = torch.tensor(sc_outp.inverse_transform(y[H]))
        reallosses = torch.tensor(y[H])
        kernel_info = [m.state_dict()]

        # Optimization loop
        for t in range(T):
            print(".",end="",flush=True)
            # run acquisition on params
            if acqfun.__name__ in ["evaluation_EI","evaluation_POI"]: # these need the best setting so far
                candidate = acqfun(m,X.unsqueeze(1),losses[-1],H=H)
            else:
                candidate = acqfun(m,X.unsqueeze(1),H=H)
    
            # Append new candidate
            losses.append(  max(y[candidate],losses[-1]  )) #append improvements
            # losses.append(  max(sc_outp.inverse_transform(y[candidate]),losses[-1]  )) #append improvements
            # reallosses = torch.cat((reallosses,torch.tensor(sc_outp.inverse_transform(y[candidate])).unsqueeze(0)),0) #append actual loss
            reallosses = torch.cat((reallosses,torch.tensor(y[candidate]).unsqueeze(0)),0) #append actual loss
            H = np.append(H,candidate.cpu()) #append chosen parameters
            
            # Fit new GP model
            m = fit_simple_custom_gp(X[H],y[H],seed)
            kernel_info.append(m.state_dict())         
            
            
        # append to history of acqfuns
        improvement_hist[acqfun.__name__] = losses
        loss_hist[acqfun.__name__] = reallosses
        param_hist[acqfun.__name__] = H
        kernel_hist[acqfun.__name__] = kernel_info
        
    return improvement_hist,loss_hist,param_hist,kernel_hist

# function to save trajectories to pkl
def save_hist(losshist,improvementhist,paramhist,kernel_hist,datasetname,spacename):
    globalhist = {}
    globalhist["loss"] = losshist
    globalhist["improvement"] = improvementhist
    globalhist["params"] = paramhist
    globalhist["kernel"] = kernel_hist
    
    with open(os.path.join(rootdir,"trajectories",'hist_'+datasetname+"_"+spacename+'.pkl'), 'wb') as handle:
        pkl.dump(globalhist, handle)
        
# Generate a SMBO sequence
def Generator(data,space):
    print("starting search space: "+space)
    if space=="hpo_data":
        datkeys = args.keys
    else:
        datkeys = list(data.keys())

    seedlist = list(np.random.randint(0,10000,2000)) # or hardcode list of seeds
    numseeds = 10
    T = 100
    
    for dataset in datkeys:
        print("Starting dataset: "+dataset,flush=True)
        if dataset in args.done:
            print("Already done")
            gc.collect()
            continue
        if args.dataset == "hpo_b" and args.only != None:
            if dataset not in args.only:
                gc.collect()
                continue
                
        losshistdataset = {}
        improvementhistdataset = {}
        paramhistdataset = {}
        kernelhistdataset = {}
        
         # load checkpoint if its available
        if os.path.exists(os.path.join(rootdir,"trajectories",'hist_'+dataset+"_"+space+'.pkl')):
            with open(os.path.join(rootdir,"trajectories",'hist_'+dataset+"_"+space+'.pkl'),"rb") as f:
                trajectories = pkl.load(f)
                losshistdataset = trajectories["loss"]
                improvementhistdataset = trajectories["improvement"]
                paramhistdataset = trajectories["params"]
                kernelhistdataset = trajectories["kernel"]       
                
                doneseeds = list(losshistdataset.keys())
                nseeds = len(doneseeds) # number of seeds already done
                numseeds = 10 - nseeds
        else:
            numseeds = 10
            nseeds = 0
            doneseeds = []
        # Load data
        x = torch.tensor(data[dataset]["X"])
        if space!="hpo_data":
            y = torch.tensor(data[dataset]["y"])
        else:
            y = torch.tensor(data[dataset]["Y"])
        seeds = seedlist.copy()
        # Use different seeds
        for i in range(numseeds):
            print("Starting seed {}".format(nseeds+i),flush=True)
            # Catch cholesky error. If error happens, restart with new seed
            cholesky = True
            num_chol = 0
            while(cholesky):
                num_chol+=1
                try:
                    seed = seeds.pop()
                    print(seed)
                    if str(seed) in doneseeds:
                        continue
                except:
                    # no seeds left, cannot compute
                    print("No seeds left",flush=True)
                    break
                # Generate Sequences with all acquisition functions
                try:
                    
                    with gpytorch.settings.cholesky_jitter(1e-1):
                        improvements,losses,params,kernels = GenerateSequence(x,y,T,seed)
                    losshistdataset[str(seed)] = losses
                    improvementhistdataset[str(seed)] = improvements
                    paramhistdataset[str(seed)] = params
                    kernelhistdataset[str(seed)] = kernels
                except Exception as e:
                    print("error occured",flush=True)
                    print(e)
                    gc.collect()
                    continue
                
                    
                cholesky = False # sucessful
            print("",flush=True)   
            print("Saving hist...",flush=True)   
            save_hist(losshistdataset,improvementhistdataset,paramhistdataset,kernelhistdataset,dataset,space)

# run Generator

# run on hpo_data
if args.dataset=="hpo_data":
    Generator(hpo_data,"hpo_data")

# run on hpo_b
if args.dataset == "hpo_b":
    for search_space in ids:
        if search_space not in args.keys:
            continue
        Generator(hpob_hdlr.meta_test_data[search_space],search_space)
    
# print runtime
print("--- Ran for %s seconds ---" % (time.time() - start_time),flush=True)