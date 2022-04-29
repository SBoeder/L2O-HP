# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:01:13 2021

@author: simon

Run baselines. This will be used by the "Tests.py" to compare against baselines.
"""
import os
import pickle as pkl
import torch
from Utility import get_hdlr,data_loader,get_seeds,compute_regret
import numpy as np
import random
from Environment import Environment

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(int(seed))
    random.seed(seed)
    
    rng = np.random.default_rng(int(seed))
    
    return rng

def create_stats(X,y,trajectories):
    max_actions = X.shape[0]-10
    traj_accs =  y[trajectories].squeeze(-1)
    regret,regret_curve = compute_regret(traj_accs,y)
    time_to_best = torch.max(traj_accs,axis=1)[1] 
    average_regret = regret[:max_actions].mean() 
    average_normalized_regret = torch.mean(regret_curve,axis=0)
    return traj_accs,time_to_best,regret_curve,average_normalized_regret,average_regret
    
def BO_objective(candidate):
    global trajectory
    candidate_dist = torch.cdist(torch.tensor(candidate).unsqueeze(0).float(), X.float())
    candidate_dist = candidate_dist.scatter(1,trajectory.unsqueeze(0),np.inf)
    candidate_idx = torch.argmin(candidate_dist,axis=1)
  
    trajectory = torch.cat((trajectory,candidate_idx))
        
    return y[candidate_idx].numpy().reshape(-1)

# Grid Search
def Grid(searchspace,testdatasets,hdlr=None,T=100):
    
    # Get hdlr if not handed over
    if hdlr==None:
        hdlr,ids = get_hdlr(searchspace)
    
    # rollout for each testdataset
    alltrajectories = {}
    for testdataset in testdatasets:
        # load data & create environment
        X,y = data_loader(searchspace,testdataset,hdlr,device=torch.device("cpu"))
        env = Environment(X,y)
        
        # get testseeds
        tseeds  = get_seeds(searchspace,testdataset)
        
        domain = torch.zeros([2,X.shape[1]])
        
        # get min/max of every feature
        domain[0] = X.min(axis=0).values
        domain[1] = X.max(axis=0).values
        
        # get linear aligned params
        n_per_col = np.ceil(np.power(100,1/X.shape[1])).astype(int)
        gridded_per_col = np.linspace(domain[0],domain[1],num=n_per_col).T
        
        # rollout for every seed
        trajectories = torch.tensor([])
        for tseed in tseeds:
            # initialize with starting points
            trajectory = torch.tensor([]).long()
            _,_,losses,idxs = env.initGP(tseed)
            trajectory = torch.cat((trajectory,torch.tensor(idxs)))
            
            # get 100 random combinations, put them to grid and evaluate
            already_chosen = []
            for t in range(T):
                while(True):
                    idx = np.random.choice(gridded_per_col.shape[1],gridded_per_col.shape[0])
                    if idx.tolist() not in already_chosen:
                        already_chosen.append(idx.tolist())
                        break
                candidate = torch.tensor(np.take(gridded_per_col,idx))
                candidate_dist = torch.cdist(candidate.unsqueeze(0), X.float())
                candidate_dist = candidate_dist.scatter(1,trajectory.unsqueeze(0),np.inf)
                candidate_idx = torch.argmin(candidate_dist,axis=1)
                trajectory = torch.cat((trajectory,candidate_idx))
        
            trajectories = torch.cat((trajectories,trajectory.unsqueeze(0)))
        traj_accs,time_to_best,regret_curve,average_normalized_regret,average_regret = create_stats(X, y, trajectories.long())
        alltrajectories[testdataset] = {'testseeds':tseeds,'trajectories':trajectories,"accs":traj_accs,"time_to_best":time_to_best,"regret_curve":regret_curve,"average_normalized_regret":average_normalized_regret,"average_regret":average_regret} 
    return alltrajectories
    
    
# Random Search
def Random(searchspace,testdatasets,hdlr=None,T=100):
    # Get hdlr if not handed over
    if hdlr==None:
        hdlr,ids = get_hdlr(searchspace)
    # rollout for each testdataset
    alltrajectories = {}
    for testdataset in testdatasets:
        # load data & create environment
        X,y = data_loader(searchspace,testdataset,hdlr,device=torch.device("cpu"))
        env = Environment(X,y)
        
        # get testseeds
        tseeds  = get_seeds(searchspace,testdataset)
                
        # rollout for every seed
        trajectories = torch.tensor([])
        for tseed in tseeds:
            trajectory = torch.tensor([]).long()
            _,_,losses,idxs = env.initGP(tseed)
           
            idxrange = np.delete(np.arange(X.shape[0]),trajectory)
            np.random.shuffle(idxrange)
            new_idxs = idxrange[:T]
            trajectory = torch.cat((trajectory,torch.tensor(idxs),torch.tensor(new_idxs)))
            trajectories = torch.cat((trajectories,trajectory.unsqueeze(0)))
        traj_accs,time_to_best,regret_curve,average_normalized_regret,average_regret = create_stats(X, y, trajectories.long())
        alltrajectories[testdataset] = {'testseeds':tseeds,'trajectories':trajectories,"accs":traj_accs,"time_to_best":time_to_best,"regret_curve":regret_curve,"average_normalized_regret":average_normalized_regret,"average_regret":average_regret} 
    return alltrajectories
             
# Bayesian Opt (UCB, EI, POI, MES)
def BO_UCB(searchspace,testdatasets,hdlr=None,T=100):
    return BO_single(searchspace,testdatasets,hdlr=None,T=100,acqf = "evaluation_UCB")
def BO_EI(searchspace,testdatasets,hdlr=None,T=100):
    return BO_single(searchspace,testdatasets,hdlr=None,T=100,acqf = "evaluation_EI")
def BO_POI(searchspace,testdatasets,hdlr=None,T=100):
    return BO_single(searchspace,testdatasets,hdlr=None,T=100,acqf = "evaluation_POI")
def BO_MES(searchspace,testdatasets,hdlr=None,T=100):
    return BO_single(searchspace,testdatasets,hdlr=None,T=100,acqf = "evaluation_MES")

def BO_single(searchspace,testdatasets,hdlr=None,T=100,acqf = None):
    if hdlr==None:
        hdlr,ids = get_hdlr(searchspace)
    if acqf == None:
        print("No acquisition function defined",flush=True)
        return
    
    # rollout for each testdataset
    alltrajectories = {}
    
    for testdataset in testdatasets:
        # load data & create environment
        X,y = data_loader(searchspace,testdataset,hdlr,device=torch.device("cpu"))
        
        # get trajectories generated
        with open("../trajectories/hist_{}_{}.pkl".format(testdataset,searchspace),"rb") as f:
            history = pkl.load(f)
        tseeds = list(history["loss"].keys())
        trajectories = torch.tensor([])
        for tseed in tseeds:
            trajectories =  torch.cat((trajectories,torch.tensor(history["params"][tseed][acqf]).unsqueeze(0))) 
        
        # get stats
        traj_accs,time_to_best,regret_curve,average_normalized_regret,average_regret = create_stats(X, y, trajectories.long())
        alltrajectories[testdataset] = {'testseeds':tseeds,'trajectories':trajectories,"accs":traj_accs,"time_to_best":time_to_best,"regret_curve":regret_curve,"average_normalized_regret":average_normalized_regret,"average_regret":average_regret} 
        
    return alltrajectories


def BO(searchspace,testdatasets,hdlr):
    # rollout for each testdataset
    alltrajectories = {}
    
    for testdataset in testdatasets:
        # load data & create environment
        X,y = data_loader(searchspace,testdataset,hdlr,device=torch.device("cpu"))
        
        # get trajectories generated
        with open("../trajectories/hist_{}_{}.pkl".format(testdataset,searchspace),"rb") as f:
            history = pkl.load(f)
        tseeds = list(history["loss"].keys())
        trajectories = torch.tensor([])
        for tseed in tseeds:
            for acqf in history["params"][tseed].keys():
                trajectories =  torch.cat((trajectories,torch.tensor(history["params"][tseed][acqf]).unsqueeze(0))) 
        
        # get stats
        traj_accs,time_to_best,regret_curve,average_normalized_regret,average_regret = create_stats(X, y, trajectories.long())
        alltrajectories[testdataset] = {'testseeds':tseeds,'trajectories':trajectories,"accs":traj_accs,"time_to_best":time_to_best,"regret_curve":regret_curve,"average_normalized_regret":average_normalized_regret,"average_regret":average_regret} 
        
    return alltrajectories

def BO2(searchspace,testdatasets,hdlr):
    # bad datasets ['272', '146082']
    return

# SMAC
class SMAC():
    def __init__(self,searchspace,testdatasets,hdlr=None,T=110):
        self.T=T
        self.searchspace=searchspace
        self.testdatasets=testdatasets
        if hdlr == None:
           self.hdlr,self.ids = get_hdlr(searchspace)
        else:
            self.hdlr = hdlr
            
        self.alltrajectories = {}
        
    def SMAC_objective(self,config):
        candidate = []
        for param in config:
            candidate.append(config[param])
        candidate = np.stack(candidate)
        candidate_dist = torch.cdist(torch.tensor(candidate).unsqueeze(0).float(), self.X.float())
        candidate_dist = candidate_dist.scatter(1,self.trajectory.unsqueeze(0),np.inf)
        candidate_idx = torch.argmin(candidate_dist,axis=1)
        self.trajectory = torch.cat((self.trajectory,candidate_idx))
            
        return self.y[candidate_idx].numpy().reshape(-1)

    def run_SMAC(self):
        from ConfigSpace import ConfigurationSpace
        from ConfigSpace.hyperparameters import UniformFloatHyperparameter
        from smac.facade.smac_bb_facade import SMAC4BB
        from smac.scenario.scenario import Scenario
        

        for testdataset in self.testdatasets:
            print("Starting dataset {}".format(testdataset),flush=True)
            # load data & create environment
            self.X,self.y = data_loader(searchspace,testdataset,hdlr,device=torch.device("cpu"))
            self.y = self.y * -1
            
            
            lb = self.X.min(axis=0).values.numpy()
            ub = self.X.max(axis=0).values.numpy()
            
            while((lb>=ub).sum()):
                delparam = np.where((lb>=ub).astype(int)>0)[0][0]
                self.X = torch.cat((self.X[:,:delparam], self.X[:,delparam+1:]),axis=1)
                lb = self.X.min(axis=0).values.numpy()
                ub = self.X.max(axis=0).values.numpy()
                print("Deleted parameter {}".format(delparam),flush=True)
                
            # create config object
            self.configspace = ConfigurationSpace()
            # colnames = ["param_{}".format(i) for i in range(self.X.shape[1])]
            for i,col in enumerate(self.X.T):
                self.configspace.add_hyperparameter(UniformFloatHyperparameter("param_{}".format(i), lower=col.min().item(), upper=col.max().item()))
                
            env = Environment(self.X,self.y)
            
            # get testseeds
            tseeds  = get_seeds(searchspace,testdataset)
            
            # create SMAC optimizer
            scenario = Scenario({
            "run_obj": "quality",  # Optimize quality 
            "runcount-limit": self.T,  # Max number of function evaluations (the more the better)
            "cs": self.configspace,
            'abort_on_first_run_crash': False
            })
         
            # rollout for every seed
            self.trajectories = torch.tensor([])
            for tseed in tseeds:
                print(".",end="",flush=True)
                self.trajectory = torch.tensor([]).long()
                _,_,losses,idxs = env.initGP(tseed)
                self.trajectory = torch.cat((self.trajectory,torch.tensor(idxs)))
                
                self.smac = SMAC4BB(scenario=scenario, tae_runner=self.SMAC_objective)
                best_found_config = self.smac.optimize()
                
                self.trajectories = torch.cat((self.trajectories,self.trajectory.unsqueeze(0)))
                
            traj_accs,time_to_best,regret_curve,average_normalized_regret,average_regret = create_stats(self.X, self.y*-1, self.trajectories.long())
            self.alltrajectories[testdataset] = {'testseeds':tseeds,'trajectories':self.trajectories,"accs":traj_accs,"time_to_best":time_to_best,"regret_curve":regret_curve,"average_normalized_regret":average_normalized_regret,"average_regret":average_regret} 
            
            # checkpoint results
            print("Saving Checkpoint",flush=True)
            with open("../teststatistics/baselines/SMAC/checkpoint_{}.pkl".format(searchspace),"wb") as f:
                pkl.dump(self.alltrajectories,f)
            
        return self.alltrajectories

# CMA-ES
def CMAES(searchspace,testdatasets,hdlr=None,T=100):
    from cmaes import CMA

    if hdlr==None:
        hdlr,ids = get_hdlr(searchspace)
    # rollout for each testdataset
    alltrajectories = {}
    for testdataset in testdatasets:
        print("Starting dataset {}".format(testdataset),flush=True)
        # load data & create environment
        X,y = data_loader(searchspace,testdataset,hdlr,device=torch.device("cpu"))
        env = Environment(X,y)
        
        # get bounds
        bounds = np.array([ X.min(axis=0).values.numpy(), X.max(axis=0).values.numpy()]).T
        
        # get testseeds
        tseeds  = get_seeds(searchspace,testdataset)
        
        # rollout for every seed
        trajectories = torch.tensor([])
        for tseed in tseeds:
            print("Starting seed {}".format(tseed),flush=True)
            # initialize with starting points
            trajectory = torch.tensor([]).long()
            _,_,losses,idxs = env.initGP(tseed)
            trajectory = torch.cat((trajectory,torch.tensor(idxs)))
            
            # create CMA-ES optimizer
            sigma = abs(bounds[:,0] - bounds[:,1]).max()  / 5 # 1/5 of domain bounds maximmum
            cma_opt = CMA(mean=np.zeros(X.shape[1]), sigma=sigma,bounds = bounds ,population_size=10)
            
            # append initial idxs to HEBO known solutions
            initial_solutions = []
            for idx in trajectory:
                initial_solutions.append((X[idx].numpy(),y[idx].numpy()))
            cma_opt.tell(initial_solutions)
            
            for t in range(int(T /10)):
                solutions = []
                for _ in range(cma_opt.population_size):
                    print(".",flush=True,end="")
                    candidate = cma_opt.ask()
                    candidate_dist = torch.cdist(torch.tensor(candidate).unsqueeze(0).float(), X.float())
                    candidate_dist = candidate_dist.scatter(1,trajectory.unsqueeze(0),np.inf)
                    candidate_idx = torch.argmin(candidate_dist,axis=1)
                    solutions.append((X[candidate_idx].numpy().reshape(-1),y[candidate_idx].numpy().reshape(-1)))
                    trajectory = torch.cat((trajectory,candidate_idx))
                cma_opt.tell(solutions)
            trajectories = torch.cat((trajectories,trajectory.unsqueeze(0)))
            
        traj_accs,time_to_best,regret_curve,average_normalized_regret,average_regret = create_stats(X, y, trajectories.long())
        alltrajectories[testdataset] = {'testseeds':tseeds,'trajectories':trajectories,"accs":traj_accs,"time_to_best":time_to_best,"regret_curve":regret_curve,"average_normalized_regret":average_normalized_regret,"average_regret":average_regret} 
        
    return alltrajectories

# BOHAMIANN
def BOHAMIANN(searchspace,testdatasets,hdlr=None,T=100):
    from robo.fmin.bayesian_optimization import bayesian_optimization
    import warnings
    warnings.filterwarnings("ignore")
    

    if hdlr==None:
        hdlr,ids = get_hdlr(searchspace)
    # rollout for each testdataset
    alltrajectories = {}
    
    # check if checkpoint available
    if os.path.isfile("../teststatistics/baselines/BOHAMIANN/checkpoint_{}.pkl".format(searchspace)):
        print("Loading checkpoint",flush=True)
        with open("../teststatistics/baselines/BOHAMIANN/checkpoint_{}.pkl".format(searchspace),"rb") as f:
            alltrajectories = pkl.load(f)
        for dataset,results in alltrajectories.items():
            testdatasets.remove(dataset)
            
    for testdataset in testdatasets:
        print("Starting dataset {}".format(testdataset),flush=True)
        # load data & create environment
        global X
        global y
        X,y = data_loader(searchspace,testdataset,hdlr,device=torch.device("cpu"))
        y = y * -1 # minimize the opposite direction      
        
        # get bounds
        lb = X.min(axis=0).values.numpy()
        ub = X.max(axis=0).values.numpy()
        
        while((lb>=ub).sum()):
            delparam = np.where((lb>=ub).astype(int)>0)[0][0]
            X = torch.cat((X[:,:delparam], X[:,delparam+1:]),axis=1)
            lb = X.min(axis=0).values.numpy()
            ub = X.max(axis=0).values.numpy()
            print("Deleted parameter {}".format(delparam),flush=True)
            
        env = Environment(X,y)
        
        # get testseeds
        tseeds  = get_seeds(searchspace,testdataset)
        
        # rollout for every seed
        trajectories = torch.tensor([])
        for tseed in tseeds:
            print("Starting seed {}".format(tseed),flush=True)
            # initialize with starting points
            global trajectory
            trajectory = torch.tensor([]).long()
            _,_,losses,idxs = env.initGP(tseed)
            trajectory = torch.cat((trajectory,torch.tensor(idxs)))
            try:
                results = bayesian_optimization(BO_objective, lb, ub, model_type="bohamiann", num_iterations=101, X_init=X[idxs].numpy(),Y_init=y[idxs].numpy(),n_init=1)
            except Exception as e:
                print("Error occured: {}".format(e),flush=True)
                trajectory = torch.cat((trajectory,trajectory[-1].repeat(110-trajectory.shape[0]))) # padding with same if sampling fails
            trajectories = torch.cat((trajectories,trajectory.unsqueeze(0)))
            
        traj_accs,time_to_best,regret_curve,average_normalized_regret,average_regret = create_stats(X, y*-1, trajectories.long())
        alltrajectories[testdataset] = {'testseeds':tseeds,'trajectories':trajectories,"accs":traj_accs,"time_to_best":time_to_best,"regret_curve":regret_curve,"average_normalized_regret":average_normalized_regret,"average_regret":average_regret} 
        
        # checkpoint results
        print("Saving Checkpoint",flush=True)
        with open("../teststatistics/baselines/BOHAMIANN/checkpoint_{}.pkl".format(searchspace),"wb") as f:
            pkl.dump(alltrajectories,f)
        
    return alltrajectories

# DNGO
def DNGO(searchspace,testdatasets,hdlr=None,T=100):
    from robo.fmin.bayesian_optimization import bayesian_optimization
    import warnings
    warnings.filterwarnings("ignore")
    

    if hdlr==None:
        hdlr,ids = get_hdlr(searchspace)
    # rollout for each testdataset
    alltrajectories = {}
    
    # check if checkpoint available
    if os.path.isfile("../teststatistics/baselines/DNGO/checkpoint_{}.pkl".format(searchspace)):
        print("Loading checkpoint",flush=True)
        with open("../teststatistics/baselines/DNGO/checkpoint_{}.pkl".format(searchspace),"rb") as f:
            alltrajectories = pkl.load(f)
        for dataset,results in alltrajectories.items():
            testdatasets.remove(dataset)
            
    for testdataset in testdatasets:
        print("Starting dataset {}".format(testdataset),flush=True)
        # load data & create environment
        global X
        global y
        X,y = data_loader(searchspace,testdataset,hdlr,device=torch.device("cpu"))
        y = y * -1 # minimize the opposite direction
        
        
        # get bounds
        lb = X.min(axis=0).values.numpy()
        ub = X.max(axis=0).values.numpy()
        
        while((lb>=ub).sum()):
            delparam = np.where((lb>=ub).astype(int)>0)[0][0]
            X = torch.cat((X[:,:delparam], X[:,delparam+1:]),axis=1)
            lb = X.min(axis=0).values.numpy()
            ub = X.max(axis=0).values.numpy()
            print("Deleted parameter {}".format(delparam),flush=True)
            
        env = Environment(X,y)
        # get testseeds
        tseeds  = get_seeds(searchspace,testdataset)
        
        # rollout for every seed
        trajectories = torch.tensor([])
        for tseed in tseeds:
            print("Starting seed {}".format(tseed),flush=True)
            # initialize with starting points
            global trajectory
            trajectory = torch.tensor([]).long()
            _,_,losses,idxs = env.initGP(tseed)
            trajectory = torch.cat((trajectory,torch.tensor(idxs)))
            results = bayesian_optimization(BO_objective, lb, ub, model_type="dngo", num_iterations=101, X_init=X[idxs].numpy(),Y_init=y[idxs].numpy().reshape(-1),n_init=1)

            trajectories = torch.cat((trajectories,trajectory.unsqueeze(0)))
            
        
        traj_accs,time_to_best,regret_curve,average_normalized_regret,average_regret = create_stats(X, y*-1, trajectories.long())
        alltrajectories[testdataset] = {'testseeds':tseeds,'trajectories':trajectories,"accs":traj_accs,"time_to_best":time_to_best,"regret_curve":regret_curve,"average_normalized_regret":average_normalized_regret,"average_regret":average_regret} 
        
        # checkpoint results
        print("Saving Checkpoint",flush=True)
        with open("../teststatistics/baselines/DNGO/checkpoint_{}.pkl".format(searchspace),"wb") as f:
            pkl.dump(alltrajectories,f)
        
    return alltrajectories

# HEBO (maybe)
def HEBO(searchspace,testdatasets,hdlr=None,T=100):
    # import necessary libraries
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO
    import pandas as pd
    
    if hdlr==None:
        hdlr,ids = get_hdlr(searchspace)
    # rollout for each testdataset
    alltrajectories = {}
    
    # check if checkpoint available
    if os.path.isfile("../teststatistics/baselines/HEBO/checkpoint_{}.pkl".format(searchspace)):
        print("Loading checkpoint",flush=True)
        with open("../teststatistics/baselines/HEBO/checkpoint_{}.pkl".format(searchspace),"rb") as f:
            alltrajectories = pkl.load(f)
        for dataset,results in alltrajectories.items():
            testdatasets.remove(dataset)
    
    for testdataset in testdatasets:
        print("Starting dataset {}".format(testdataset),flush=True)
        # load data & create environment
        X,y = data_loader(searchspace,testdataset,hdlr,device=torch.device("cpu"))
        # get testseeds
        tseeds  = get_seeds(searchspace,testdataset)
        
        # create HEBO optimizer space
        spacelist = []
        while True:
            delind = -1
            for i,col in enumerate(X.T):
                if col.min().item() == col.max().item(): # when the lower and upper bound is the same
                   delind = i
                   break
            if delind==-1:
                break
            else:
                X = np.delete(X,delind,1)
        
        for i,col in enumerate(X.T): #now iterate over new X and add spaces
            spacelist.append({'name':'param_{}'.format(i),'type':'num','lb':col.min().item(),'ub':col.max().item()})
            
        space = DesignSpace().parse(spacelist)
        env = Environment(X,y)    
        colnames = ["param_{}".format(i) for i in range(X.shape[1])]
        
        # rollout for every seed
        trajectories = torch.tensor([])
        for tseed in tseeds:
            print("Starting seed {}".format(tseed),flush=True)
            # create HEBO optimizer
            hebo_opt = HEBO(space, model_name = 'gpy', rand_sample = 1)
            # initialize with starting points
            trajectory = torch.tensor([]).long()
            _,_,losses,idxs = env.initGP(tseed)
            trajectory = torch.cat((trajectory,torch.tensor(idxs)))
            # append initial idxs to HEBO known solutions
            hebo_opt.observe(pd.DataFrame(X[trajectory].numpy(),columns=colnames), y[trajectory].numpy())
        
            for t in range(T):
                print(".",flush=True,end="")
                candidate = hebo_opt.suggest(n_suggestions=1)
                candidate_dist = torch.cdist(torch.tensor(candidate.to_numpy()).float(), X)
                candidate_dist = candidate_dist.scatter(1,trajectory.unsqueeze(0),np.inf)
                candidate_idx = torch.argmin(candidate_dist,axis=1)
                hebo_opt.observe(candidate, y[candidate_idx].numpy())
                trajectory = torch.cat((trajectory,candidate_idx))
            trajectories = torch.cat((trajectories,trajectory.unsqueeze(0)))
            
        traj_accs,time_to_best,regret_curve,average_normalized_regret,average_regret = create_stats(X, y, trajectories.long())
        alltrajectories[testdataset] = {'testseeds':tseeds,'trajectories':trajectories,"accs":traj_accs,"time_to_best":time_to_best,"regret_curve":regret_curve,"average_normalized_regret":average_normalized_regret,"average_regret":average_regret} 
        
        # checkpoint results
        print("Saving Checkpoint",flush=True)
        with open("../teststatistics/baselines/HEBO/checkpoint_{}.pkl".format(searchspace),"wb") as f:
            pkl.dump(alltrajectories,f)
                 
    return alltrajectories

def L2OHP(searchspace,testdatasets,hdlr=None,T=100,dagger=False):
    from Models import Learner
    from Agent import Agent
    from Environment import Environment
    from Utility import create_pairs
    import configparser
  
    device = torch.device("cuda")
    if hdlr==None:
        hdlr,ids = get_hdlr(searchspace)
    # rollout for each testdataset
    alltrajectories = {}
    
    if dagger:
        postfix = "_dagger"
    else:
        postfix = ""
        
    config = configparser.ConfigParser()
    config.read("../model_configs/{}.ini".format(0))
    
    # load L2O-HP model
    state_size,action_size = create_pairs(searchspace,testdatasets,hdlr,get_state_size=True)
   
    model = Learner(state_size,action_size,
                    hidden_size=config.getint("DEFAULT","hidden_size"),
                    transformer_hidden_size=config.getint("DEFAULT","transformer_hidden_size"),
                    transformer_out_features=config.getint("DEFAULT","transformer_out_features"),
                    heads=4,
                    n_hidden_layers=config.getint("DEFAULT","n_hidden_layers")
                    ).to(device=device)
    
    model.load_state_dict(torch.load("../models/tests/learner_{}{}_test.pt".format(searchspace,postfix)))
   
    # check if checkpoint available
    if os.path.isfile("../teststatistics/baselines/L2OHP/checkpoint_{}.pkl".format(searchspace)):
        print("Loading checkpoint",flush=True)
        with open("../teststatistics/baselines/L2OHP/checkpoint_{}.pkl".format(searchspace),"rb") as f:
            alltrajectories = pkl.load(f)
        for dataset,results in alltrajectories.items():
            testdatasets.remove(dataset)
         
    for testdataset in testdatasets:
        print("Starting dataset {}".format(testdataset),flush=True)
        # load data & create environment
        X,y = data_loader(searchspace,testdataset,hdlr,device=torch.device("cuda"))
        # get testseeds
        tseeds  = get_seeds(searchspace,testdataset)
        
        agent = Agent(model)
        environment = Environment(X.float(),y.float())
                
        # evaluate performance for every seed
        trajectories,_,_ = agent.rollout(environment,100,tseeds,device=torch.device("cuda")) # return traj per seed to make it comparable
        traj_accs,time_to_best,regret_curve,average_normalized_regret,average_regret = create_stats(X, y, trajectories.long())
        alltrajectories[testdataset] = {'testseeds':tseeds,'trajectories':trajectories,"accs":traj_accs,"time_to_best":time_to_best,"regret_curve":regret_curve,"average_normalized_regret":average_normalized_regret,"average_regret":average_regret} 
        
        # checkpoint results
        print("Saving Checkpoint",flush=True)
        with open("../teststatistics/baselines/L2OHP/checkpoint_{}.pkl".format(searchspace),"wb") as f:
            pkl.dump(alltrajectories,f)
            
    return alltrajectories

# If used to generate baseline trajectories
if __name__ == "__main__":
    import argparse
    from Utility import get_testdatasets
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',"--searchspace",nargs="*")
    parser.add_argument("-a","--algorithms",nargs="*",default=["Grid"],type=str)
    args = parser.parse_args()
    print("arguments read...",flush=True)
    
    for searchspace in args.searchspace:
        print("Searchspace: "+searchspace,flush=True)
        hdlr,ids = get_hdlr(searchspace)
            
        # get the testdatasets used in the rollouts
        testdatasets = get_testdatasets(searchspace)
        
        # run the baseline for each testdataset
        for algorithm in args.algorithms:
            print("Algorithm: {}".format(algorithm),flush=True)
            
            # run selected algorithm and create stats, do by class if SMAC
            if algorithm == "SMAC":
                allstats = SMAC(searchspace,testdatasets,hdlr=hdlr).run_SMAC()
            else:
                allstats = globals()[algorithm](searchspace,testdatasets,hdlr=hdlr)
            
            # save stats
            rootpath = "../teststatistics/baselines/{}".format(algorithm)
            if not os.path.isdir(rootpath):
                os.mkdir(rootpath)
            filepath = rootpath + "/tests_{}.pkl".format(searchspace)
            with open(filepath, "wb") as f:
                pkl.dump(allstats, f)
    