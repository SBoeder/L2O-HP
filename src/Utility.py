# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:00:28 2021

@author: simon

All utility functions 
"""
import torch
#from botorch.models.gpytorch import GPyTorchModel
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim.fit import fit_gpytorch_torch
from HPO_B import hpob_handler
import pickle as pkl
import os 
import pandas as pd
import numpy as np
from botorch.acquisition.analytic import UpperConfidenceBound, ProbabilityOfImprovement,ExpectedImprovement
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy



# Define fit function
def fit_simple_custom_gp(Xs, Ys,seed,device=torch.device("cpu")):
    # runs faster on cpu
    torch.manual_seed(seed)
    model = SingleTaskGP(Xs, Ys).to(device)
    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(device)
    mll.train()
    fit_gpytorch_torch(mll,track_iterations =False)
    mll.eval()
    return model

def load_GP(Xs,Ys,state_dict,seed):
    torch.manual_seed(seed)
    model = SingleTaskGP(Xs, Ys).to(device=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Define EI evaluation function, which in this case returns real response from metadata set
def evaluation_EI(model,data,best_f,H=None,batch_size = 256):
    model.eval()
    model.likelihood.eval()
    EI = ExpectedImprovement(model,best_f=best_f)
    n_batches = np.ceil(data.shape[0] / batch_size).astype(int)
    allresults = torch.tensor([])
    for batch in range(n_batches):
        EI_range = EI(data[batch*batch_size:(batch+1)*batch_size]).cpu()
        allresults = torch.cat((allresults,EI_range))    
    
    del EI
    # generate candidate
    if H is not None:
        allresults[H]= -np.inf
        return torch.argmax(allresults)
    else:
        return allresults
    
# Define UCB evaluation function, which in this case returns real response from metadata set
def evaluation_UCB(model,data,best_f,H=None,b=0.2,batch_size = 256):
    model.eval()
    model.likelihood.eval()
    UCB = UpperConfidenceBound(model, beta=b)
    n_batches = np.ceil(data.shape[0] / batch_size).astype(int)
    allresults = torch.tensor([])
    for batch in range(n_batches):
        UCB_range = UCB(data[batch*batch_size:(batch+1)*batch_size]).cpu()
        allresults = torch.cat((allresults,UCB_range)) 
    del UCB
    # generate candidate
    if H is not None:
        allresults[H]= -np.inf
        return torch.argmax(allresults)
    else:
        return allresults
    
def evaluation_MES(model,data,best_f,H=None,batch_size = 256):
    model.eval()
    model.likelihood.eval()
    MES = qMaxValueEntropy(model,data.squeeze(1))
    n_batches = np.ceil(data.shape[0] / batch_size).astype(int)
    allresults = torch.tensor([])
    for batch in range(n_batches):
        mes = MES(data[batch*batch_size:(batch+1)*batch_size]).cpu()
        allresults = torch.cat((allresults,mes))
    del MES
    # generate candidate
    if H is not None:
        allresults[H]= -np.inf
        return torch.argmax(allresults)
    else:
        return allresults
    
# probability of improvement
def evaluation_POI(model,data,best_f,H=None,batch_size = 256):
    model.eval()
    model.likelihood.eval()
    POI = ProbabilityOfImprovement(model,best_f=best_f)
    n_batches = np.ceil(data.shape[0] / batch_size).astype(int)
    allresults = torch.tensor([])
    for batch in range(n_batches):
        POI_range = POI(data[batch*batch_size:(batch+1)*batch_size]).cpu()
        allresults = torch.cat((allresults,POI_range))
    del POI
    # generate candidate
    if H is not None:
        allresults[H]= -np.inf
        return torch.argmax(allresults)
    else:
        return allresults

def get_expert_actions(GP_model,data,best_f,history,device=torch.device("cpu"),batch_size = 256):
    acqfs = [evaluation_EI,evaluation_UCB,evaluation_MES,evaluation_POI]
    actions = torch.tensor([])
    # action_dict = {}
    GP_model = GP_model.to(device)
    GP_model.likelihood = GP_model.likelihood.to(device)
    data = data.to(device)
    # history = history.to(device)
    for acqf in acqfs:
        try:
            with torch.no_grad():
                action_result = acqf(GP_model,data,best_f,history,batch_size=batch_size)
                actions = torch.cat((actions,action_result.unsqueeze(0)))
                # action_dict[acqf.__name__] = action_result.long()
        except Exception as e:
            print(e)
            continue
    return actions.long()
    # try:
    #     acqf_EI = evaluation_EI(GP_model,data,best_f,history)
    # try:
    #     acqf_UCB = evaluation_UCB(GP_model,data,H=history)
    # try:
    #     acqf_MES = evaluation_MES(GP_model,data,history)
    # acqf_POI = evaluation_POI(GP_model,data,best_f,history)
       
    # return torch.stack((acqf_EI,acqf_UCB,acqf_POI)),{"acqf_EI":acqf_EI,"acqf_UCB":acqf_UCB,"acqf_POI":acqf_POI}

# ------------  NOT NEEDED ANYMORE START --------------
def quickfix(trajectory,searchspace,dataset,data,seed,device):  
    '''Inplace operation!'''
    if searchspace != "hpo_data":
        x = torch.tensor(data.meta_test_data[searchspace][dataset]["X"])
        y = torch.tensor(data.meta_test_data[searchspace][dataset]["y"])
    else:
        x = torch.tensor(data[dataset]["X"])
        y = torch.tensor(data[dataset]["Y"])
    initidx = trajectory["params"][seed]["evaluation_UCB"][:10]
    initkernel = fit_simple_custom_gp(x[initidx],y[initidx],int(seed),device=torch.device("cpu"))
    for acqf in list(trajectory["kernel"][seed].keys()):
        trajectory["kernel"][seed][acqf].insert(0,initkernel.state_dict())
        
# ------------  NOT NEEDED ANYMORE END --------------

def data_loader(searchspace,dataset,hdlr=None,device=torch.device("cuda")):
    if searchspace == "hpo_data":
        if hdlr == None:
            rootdir = "../"
            data_file = os.path.join(rootdir, "AA.pkl")
            with open(data_file, "rb") as f:
                hdlr = pkl.load(f)
        x = torch.tensor(hdlr[dataset]["X"]).to(device=device)
        y = torch.tensor(hdlr[dataset]["Y"]).to(device=device)
    else:
        if hdlr==None:
            hdlr = hpob_handler.HPOBHandler(root_dir="../hpob-data/", mode="v2")
        x = torch.tensor(hdlr.meta_test_data[searchspace][dataset]["X"]).to(device=device)
        y = torch.tensor(hdlr.meta_test_data[searchspace][dataset]["y"]).to(device=device)
    return x,y

def get_keys(searchspace):
    
    if searchspace=="hpo_data":
        data_file = os.path.join("..", "AA.pkl")
        with open(data_file, "rb") as f:
            hpo_data = pkl.load(f)
        return list(hpo_data.key())
    else:
        hpob_hdlr = hpob_handler.HPOBHandler(root_dir="../hpob-data/", mode="v2")
        ids = hpob_hdlr.get_datasets(searchspace)
        return ids

def get_seeds(searchspace,dataset):
    # returns all seeds for given searchspace and dataset that have been evaluated
    with open("../trajectories/hist_{}_{}.pkl".format(dataset,searchspace),"rb") as f:
        trajectory = pkl.load(f)
    seeds = list(trajectory["loss"].keys())
    return seeds


def compute_regret(traj_accs,y):
    # Computes average normalized regret over the trajectory
    regret_at_t = (torch.max(y)- traj_accs) / (torch.max(y) - torch.min(y)) 
    
    regret_curve = torch.clone(regret_at_t) # normalized regret up until t
    for s in range(regret_at_t.shape[0]):
        for i in range(regret_at_t[s].shape[0]):
            regret_curve[s][i] = min(regret_at_t[s][:i+1])
                       
    return regret_at_t,regret_curve

def get_testdatasets(searchspace):
    with open("../teststatistics/tests_{}.pkl".format(searchspace),"rb") as f:
        results = pkl.load(f)
    return results["testdatasets"]

def get_baseline_trajectory(searchspace,dataset,device=torch.device("cuda")):
    with open("../trajectories/hist_{}_{}.pkl".format(dataset,searchspace),"rb") as f:
        trajectory = pkl.load(f)
    return trajectory["params"]
    # return torch.tensor(list(trajectory["params"][seed].values()),device=device)
    
def get_baseline_performance(searchspace,datasets,hdlr,mode="agg",device=torch.device("cuda")):
    # gets the aggregated or per dataset performance of the baseline on given datasets of given searchspace
    
    # mode = "agg" --> average over all datasets
    # mode = "per" --> perfomance per dataset TODO
    
    regrets = torch.tensor([],device=device)
    regrets_curved = torch.tensor([],device=device)
    normalized_regrets = torch.tensor([],device=device)
    normalized_regrets_dict = {}
    
    for dataset in list(datasets):
        if searchspace == "5859" and dataset in ["272","146082"]:
            continue
        X,y = data_loader(searchspace, dataset,hdlr=hdlr,device=device)
        seeds = get_seeds(searchspace, dataset)
        all_traj_accs_b = torch.tensor([],device=device)
        trajectory_b = get_baseline_trajectory(searchspace,dataset,device=device)
        for seed in seeds:
            traj_acc_b = y[torch.tensor(list(trajectory_b[seed].values()))].squeeze(-1)
            traj_acc_b = torch.mean(traj_acc_b,axis=0)
            all_traj_accs_b = torch.cat((all_traj_accs_b,traj_acc_b.unsqueeze(0)))
        
        regret_b,regret_curve_b = compute_regret(all_traj_accs_b,y)      
        regrets = torch.cat((regrets,regret_b)) # append regrets of all datasets and seeds to a large tensor
        normalized_regrets = torch.cat((normalized_regrets,regret_curve_b))
        normalized_regrets_dict[dataset] = regret_curve_b.mean(0)
        
    # average the regrets over all datasets, so we have the average normalized regret for every timestep t
    average_normalized_regret_at_at = torch.mean(normalized_regrets,axis=0)
    return average_normalized_regret_at_at,normalized_regrets, normalized_regrets_dict
    
def compute_ranks(baseline,actual,returnallranks=False):
    # TODO recode for any number of baselines, and use accuracy and not regret
    allranks = torch.tensor([])
    for traj in range(baseline.shape[0]):
        compare_df = pd.DataFrame({'baseline':baseline[traj].cpu().numpy(),'actual':actual[traj].cpu().numpy() }) # lowest number has highest rank
        ranks = torch.tensor(compare_df.rank(axis=1).values)
        allranks = torch.cat((allranks,ranks.unsqueeze(0)))
    mean_ranks = torch.mean(allranks,axis=0)
    if returnallranks:
        return mean_ranks, allranks
    return mean_ranks

def get_hdlr(searchspace):
     # Load data
    if searchspace != "hpo_data":
        hdlr = hpob_handler.HPOBHandler(root_dir="../hpob-data/", mode="v2")
        ids = hdlr.get_datasets(searchspace)
    else:
        rootdir = "../"
        data_file = os.path.join(rootdir, "AA.pkl")
        with open(data_file, "rb") as f:
            hdlr = pkl.load(f)
        ids = list(hdlr.keys())
    return hdlr,ids
    

def create_pairs(searchspace,datasets,hdlr,delta_next_action=True,T_init=100,n=10,union=True,device=torch.device("cuda"),get_state_size=False):
    '''Creates and returns state-action pairs for a given trajectory.'''
    
    # delta_next_action --> target action is the change of parameters, not the actual parameter directly
    # union --> If True, states of all datasets are gathered together. If False, the state action pairs are separated by dataset (for easier fold concatenation)
    
     # define state-action dict (two empty tensors for states and actions, for every history length 'k')
     
    if union:
        state_action_pairs = {k:(torch.tensor([],device=device),torch.tensor([],device=device)) for k in range(T_init)}
    else:
        state_action_pairs = {dataset:{k:(torch.tensor([],device=device),torch.tensor([],device=device)) for k in range(T_init)} for dataset in datasets}
    
    for dataset in datasets:
        
        with open("../trajectories/hist_{}_{}.pkl".format(dataset,searchspace),"rb") as f:
            trajectory = pkl.load(f)
            
        
        seeds = list(trajectory["loss"].keys())
        
       
        if searchspace != "hpo_data":
            data = torch.tensor(hdlr.meta_test_data[searchspace][dataset]["X"])
            y = torch.tensor(hdlr.meta_test_data[searchspace][dataset]["y"])
        else:
            data = torch.tensor(hdlr[dataset]["X"])
            y = torch.tensor(hdlr[dataset]["Y"])
            
        T = min(len(trajectory["params"][seeds[0]]["evaluation_UCB"])-n+1,y.shape[0]-n+1) # with 100 kernels (last one is missing)
                
    
        # loop over trajectories and create state-action pairs        
        for seed in seeds:        
            
            for acqf in list(trajectory["kernel"][seed].keys()):
                for t in range(T):
                    
                    # Last kernels etc should be irrelevant
                    if t==(T-1):
                        # Stop here, only max 100 actions available"
                        break
                        
                    kernels = trajectory["kernel"][seed][acqf][:t+1] # last t steps 
                    kernels = torch.stack([torch.cat((kernel['mean_module.constant'].cpu(),kernel['covar_module.raw_outputscale'].view(1).cpu(),kernel['covar_module.base_kernel.raw_lengthscale'].flatten().cpu())) for kernel in kernels])
                    kernels = torch.cat((kernels[0].repeat(n-1,1),kernels)) # same padding, so the size is the same as all other state components
                    
                    parameters = data[trajectory["params"][seed][acqf][:t+n]]
    
                    losses = trajectory["loss"][seed][acqf][:t+n]

                    action = data[trajectory["params"][seed][acqf][t+n]]
            
                    if delta_next_action:
                        # implement
                        current_position = parameters[-1]
                        action  = (action - current_position).to(device=device)
                        
                    state = torch.cat((kernels,parameters,losses),1).to(device=device)
                    if union:
                        state_action_pairs[state.shape[0]-n] = (torch.cat((state_action_pairs[state.shape[0]-n][0],state.unsqueeze(0))),torch.cat((state_action_pairs[state.shape[0]-n][1],action.unsqueeze(0))))
                    else:
                        state_action_pairs[dataset][state.shape[0]-n] = (torch.cat((state_action_pairs[dataset][state.shape[0]-n][0],state.unsqueeze(0))),torch.cat((state_action_pairs[dataset][state.shape[0]-n][1],action.unsqueeze(0))))
                    
                    if get_state_size:
                        return state_action_pairs[0][0].shape[-1],state_action_pairs[0][1].shape[-1]
                
    if union:
         return state_action_pairs,state_action_pairs[0][0].shape[-1],state_action_pairs[0][1].shape[-1]
    else:
         return state_action_pairs,state_action_pairs[datasets[0]][0][0].shape[-1],state_action_pairs[datasets[0]][0][1].shape[-1]