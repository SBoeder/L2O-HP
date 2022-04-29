# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 12:55:07 2021

@author: simon

Training functions to train a GPS model for HP optimization
"""

import torch 
from Utility import create_pairs,data_loader,get_seeds,compute_regret,get_baseline_performance,compute_ranks
import torch.nn as nn
from sklearn.model_selection import KFold
from Models import Learner
import sys
import os
from HPO_B import hpob_handler
import pickle as pkl
from Agent import Agent
from Environment import Environment
import shutil
from History_Increasing_Dataset import History_Increasing_Dataset
import gc
#%%
class Trainer():
    def __init__(self,searchspace,datasets,hidden_size=128,transformer_hidden_size=256,transformer_out_features=64,n_hidden_layers=1,heads=4,lr=1e-4,seed=42,batch_size=256):
        self.searchspace = searchspace
        self.datasets = datasets
        self.hidden_size = hidden_size
        self.transformer_hidden_size = transformer_hidden_size
        self.transformer_out_features = transformer_out_features
        self.n_hidden_layers = n_hidden_layers
        self.heads = heads
        self.lr = lr
        self.seed = seed
        self.n_hidden_layers = n_hidden_layers
        self.batch_size = batch_size
    
    #%%
    def train_learner_folds(self,epochs=200,test_interval=100,folds=5, device=torch.device("cuda")):
        '''Learner train function on k-fold CV to test imitation learning performance'''
            
        # Load data
        if self.searchspace != "hpo_data":
            self.hdlr = hpob_handler.HPOBHandler(root_dir="../hpob-data/", mode="v2")
            ids = self.hdlr.get_datasets(self.searchspace)
        else:
            rootdir = "../"
            data_file = os.path.join(rootdir, "AA.pkl")
            with open(data_file, "rb") as f:
                self.hdlr = pkl.load(f)
            ids = list(self.hdlr.keys())
            
        # create new folder for this run if needed
        path = "../models/folds/hs{}_ths{}_nhidden{}_lr{}".format(self.hidden_size,self.transformer_hidden_size,self.n_hidden_layers,self.lr)
        if not os.path.isdir(path):
            os.mkdir(path)
        # create new folder for this searchspace
        subpath = path+"/{}".format(self.searchspace)
        if not os.path.isdir(subpath):
            os.mkdir(subpath)
    
        # Arguments
        if(self.datasets=="0"):
            # read all datasets
            self.datasets = ids
            
        self.datasets = [d for d in self.datasets if os.path.isfile("../trajectories/hist_{}_{}.pkl".format(d,self.searchspace))] 
        
        if len(self.datasets)<folds:
            folds = len(self.datasets)
            
        # fold results
        torch.manual_seed(self.seed)
        results={f:(torch.tensor([]),torch.tensor([])) for f in range(folds)}
        
        # define loss function
        loss_fn = nn.L1Loss()
        
        # Define folds
        kfold = KFold(n_splits=folds, shuffle=True)
        
        # create all state_action pairs, split by dataset
        state_action_pairs,state_size,action_size = create_pairs(self.searchspace,self.datasets,self.hdlr,union=False,device=torch.device("cpu"))
            
    
        # Training loop
        for fold, (train_ids, test_ids) in enumerate(kfold.split(self.datasets)):
    
            print("Starting fold {}".format(fold),flush=True)
    
            
            # define states
            train_state_action_pairs = {k:(torch.tensor([],device=device),torch.tensor([],device=device)) for k in range(100)}
            test_state_action_pairs = {k:(torch.tensor([],device=device),torch.tensor([],device=device)) for k in range(100)}
            for train_id in train_ids:
                for k in range(100):
                    train_state_action_pairs[k] = (torch.cat((train_state_action_pairs[k][0],state_action_pairs[self.datasets[train_id]][k][0].to(device=device))),torch.cat((train_state_action_pairs[k][1],state_action_pairs[self.datasets[train_id]][k][1].to(device=device))))
            for test_id in test_ids:
                for k in range(100):
                    test_state_action_pairs[k] = (torch.cat((test_state_action_pairs[k][0],state_action_pairs[self.datasets[test_id]][k][0].to(device=device))),torch.cat((test_state_action_pairs[k][1],state_action_pairs[self.datasets[test_id]][k][1].to(device=device))))
                
            
            # define dataset and history sampler
            train = History_Increasing_Dataset(train_state_action_pairs)
            train_history_sampler = torch.utils.data.RandomSampler(train_state_action_pairs.keys())
            
            test = History_Increasing_Dataset(test_state_action_pairs)
            test_history_sampler = torch.utils.data.RandomSampler(test_state_action_pairs.keys())
             
            # define model and optimizer
            learner = Learner(state_size,action_size,hidden_size=self.hidden_size,transformer_hidden_size=self.transformer_hidden_size,transformer_out_features=self.transformer_out_features,heads=self.heads,n_hidden_layers=self.n_hidden_layers).to(device=device)
            optimizer = torch.optim.Adam(learner.parameters(), lr=self.lr)
            
        
            # load results of this fold if exists
            done= False
            if os.path.exists(subpath+"/learner_{}.pt".format(fold)):
                print("Loading from checkpoint...")
                learner.load_state_dict(torch.load(subpath+"/learner_{}.pt".format(fold)))
                with open(subpath+"/results.pkl","rb") as f:
                    temp_res = pkl.load(f)
                    results[fold] = temp_res[fold]
                done=True
             
            if not done:
                for epoch in range(epochs):
                    e_loss = 0
                    print('.',end="",flush=True)
                    for i in train_history_sampler:
                        train.sethistlength(i)
                        train_loader = torch.utils.data.DataLoader(train, batch_size = self.batch_size, shuffle = True)
                        
                        if len(train.dataset[i][0]) == 0: # skip this history length if there are no entries
                            continue
                        
                        for b,data in enumerate(train_loader):
                            optimizer.zero_grad()
                            
                            # Perform forward pass
                            outputs = learner(data[0].float())
                            
                            # Compute loss
                            loss = loss_fn(outputs, data[1])
                    
                            # Perform backward pass
                            loss.backward()
                    
                            # Perform optimization
                            optimizer.step()
                            
                            e_loss += loss.item() / (len(train_loader)*len(train_history_sampler))
                            
                    # print loss
                    if (epoch+1) % test_interval ==0:
                        print("Epoch {} loss: {}".format(epoch+1,e_loss))
                        sys.stdout.flush()       
                        
                    # start tests
                    if (epoch+1) % test_interval == 0:
                        # Validate model on test data after training
                        print("Starting tests...",flush=True)
                        cum_loss =  0
                        with torch.no_grad():
                            # Iterate over the test data and generate predictions
                            for i in test_history_sampler:
                                if len(test.dataset[i][0]) == 0:
                                    continue
                                test.sethistlength(i)
                                test_loader = torch.utils.data.DataLoader(test, batch_size = self.batch_size, shuffle = True)
                            
                                for b, data in enumerate(test_loader):
                
                                    outputs = learner(data[0].float())
                    
                                    # add loss
                                    loss = loss_fn(outputs,data[1])
                                    cum_loss += loss.item() / (len(test_loader)*len(test_history_sampler))
                        
                        print("Test loss epoch {}: ".format(epoch+1),cum_loss)
                        print("-------------------")
                    
                        # append loss to results
                        results[fold] = ( torch.cat((results[fold][0],torch.tensor(e_loss).unsqueeze(0))), torch.cat((results[fold][1],torch.tensor(cum_loss).unsqueeze(0))) )
                    
                    
                # Save model
                torch.save(learner.state_dict(),subpath+"/learner_{}.pt".format(fold))
            
            # write history file
            if not done:
                with open(subpath+"/results.pkl","wb") as f:
                    pkl.dump(results,f)
                
        return results
    
    #%%
    def train_learner_rollout(self,epochs,testinterval=0,testdatasets=0,n=10,dagger_epochs=0, do_rollout=True, device=torch.device("cuda")):
        '''Learner train function with rollout testing if needed'''
        
        # Load data
        if self.searchspace != "hpo_data":
            self.hdlr = hpob_handler.HPOBHandler(root_dir="../hpob-data/", mode="v2")
            ids = self.hdlr.get_datasets(self.searchspace)
        else:
            rootdir = "../"
            data_file = os.path.join(rootdir, "AA.pkl")
            with open(data_file, "rb") as f:
                self.hdlr = pkl.load(f)
            ids = list(self.hdlr.keys())
        
        # Arguments
        if(self.datasets=="0"):
            # read all datasets
            self.datasets = ids
            
        self.datasets = [d for d in self.datasets if os.path.isfile("../trajectories/hist_{}_{}.pkl".format(d,self.searchspace))] 
        
     
        # --> split datasets into train test    
        if testdatasets==0:
            # just take the first 30% datasets as testset if not set specifically. Take at least 1 dataset
            n_testdatasets = max(1,int(len(self.datasets) * 0.3))
            self.testdatasets = self.datasets[:n_testdatasets]
            self.datasets = self.datasets[n_testdatasets:]
        
      
        print("Creating state-action pairs...",flush=True)
        self.state_action_pairs,self.state_size,self.action_size = create_pairs(self.searchspace,self.datasets,self.hdlr,device=device)
    
        # define loss function
        self.loss_fn = nn.L1Loss()
                
        # define dataset and history sampler
        self.train = History_Increasing_Dataset(self.state_action_pairs)
        self.history_sampler = torch.utils.data.RandomSampler(self.state_action_pairs.keys())
         
        # define model and optimizer
        self.learner = Learner(self.state_size,self.action_size,hidden_size=self.hidden_size,transformer_hidden_size=self.transformer_hidden_size,transformer_out_features=self.transformer_out_features,heads=self.heads,n_hidden_layers=self.n_hidden_layers).to(device=device)
        self.optimizer = torch.optim.Adam(self.learner.parameters(), lr=self.lr)
            
        # define test statistics
        self.traj_hist = {}
        self.traj_accs_hist = {}
        self.time_to_best_hist = {}
        self.regret_hist = {}
        self.regret_curve_hist ={}
        self.average_regret_hist = {}
        
        for testset in self.testdatasets:
            self.traj_hist[testset] = torch.tensor([],device=torch.device("cpu"))#torch.tensor([],device=device)
            self.traj_accs_hist[testset] = torch.tensor([],device=torch.device("cpu"))#torch.tensor([],device=device)
            self.time_to_best_hist[testset] =torch.tensor([],device=torch.device("cpu"))# torch.tensor([],device=device)
            self.regret_hist[testset] =torch.tensor([],device=torch.device("cpu"))# torch.tensor([],device=device)
            self.regret_curve_hist[testset] =torch.tensor([],device=torch.device("cpu"))# torch.tensor([],device=device)
            self.average_regret_hist[testset] = torch.tensor([],device=torch.device("cpu"))#torch.tensor([],device=device)
            
        # Get baseline performance on testdatasets
        self.baseline_average_normalized_regret,self.baseline_normalized_regrets = get_baseline_performance(self.searchspace,self.testdatasets,self.hdlr,device=torch.device("cpu"))
        
        # gloabl test statistics
        self.average_normalized_regret_hist = torch.tensor([],device=torch.device("cpu"))#torch.tensor([],device=device)
        self.average_rank_hist = torch.tensor([],device=torch.device("cpu"))#torch.tensor([],device=device)
        
        # Load checkpoint if available
        if dagger_epochs>0:
            self.postfix = "_dagger"
        else:
            self.postfix = ""
        epochs_low=0
        if os.path.isdir("../models/checkpoints/{}{}".format(self.searchspace,self.postfix)):
            allcheckpoints = [f for f in os.listdir("../models/checkpoints/{}{}".format(self.searchspace,self.postfix)) if f.endswith("stats.pkl")]
            for checkpoint in allcheckpoints:
                point = checkpoint.replace("_stats.pkl","")
                if int(point)>epochs_low:
                    epochs_low = int(point)
            if epochs_low >0:
                print("Loading checkpoint from epoch {}".format(epochs_low))
                self.learner.load_state_dict(torch.load("../models/checkpoints/{}{}/{}.pt".format(self.searchspace,self.postfix,epochs_low)))
                self.optimizer.load_state_dict(torch.load("../models/checkpoints/{}{}/{}_optim.pt".format(self.searchspace,self.postfix,epochs_low)))
                
                # load test statistics checkpoint
                with open("../models/checkpoints/{}{}/{}_stats.pkl".format(self.searchspace,self.postfix,epochs_low),"rb") as f:
                    checkpoint_results = pkl.load(f)
                self.average_normalized_regret_hist = checkpoint_results["average_normalized_regret_hist"]
                self.average_rank_hist = checkpoint_results["average_rank_hist"]
                self.traj_hist = checkpoint_results["traj_hist"]
                self.traj_accs_hist = checkpoint_results["traj_accs_hist"]
                self.time_to_best_hist = checkpoint_results["time_to_best_hist"]
                self.regret_hist = checkpoint_results["regret_hist"]
                self.regret_curve_hist = checkpoint_results["regret_curve_hist"]
                self.average_regret_hist = checkpoint_results["average_regret_hist"]
        else:
            # if no checkpoint yet for dagger, then take the 10k model and finish imitation learning
            if dagger_epochs>0:
                # self.learner.load_state_dict(torch.load("../models/tests/learner_{}_test.pt".format(self.searchspace)))
                epochs_low = epochs
                
        # if not using dagger, train model
        self.imitation_learning(epochs_low, epochs, testinterval,do_rollout=do_rollout,device=device)
                   
        # retraining with DAgger algorithm if chosen
        if dagger_epochs > 0:
            print()
            print("Starting DAgger retraining",flush=True)
            
            # free  memory of state_action pairs
            del self.state_action_pairs
            # torch.cuda.empty_cache()
            state_action_pairs_dagger = {k:(torch.tensor([]),torch.tensor([])) for k in range(100)} # max history length is 100
            
            alldaggerstates = [int(f.replace("state_action_pairs", "").replace(".pkl","")) for f in os.listdir("../models/checkpoints/{}{}".format(self.searchspace,self.postfix)) if "state_action_pairs" in f]
            if len(alldaggerstates) >0:
                dagger_epoch_low = max(alldaggerstates)
                print("Loading from checkpoint...",flush=True)
                with open("../models/checkpoints/{}{}/state_action_pairs{}.pkl".format(self.searchspace,self.postfix,dagger_epoch_low),"rb") as f:
                    state_action_pairs_dagger = pkl.load(f)
            else:
                dagger_epoch_low = 0
                    
                    
            for dagger_epoch in range(dagger_epoch_low,dagger_epochs):
                torch.cuda.empty_cache()
                print('.',end="",flush=True)
                # rollout with current policy for each train set and save the actions the expert wouldve taken 
                for train_set in self.datasets:
                    # load data for testset
                    X,y = data_loader(self.searchspace,train_set,self.hdlr,device=device)
                    
                    # Create Agent and Environment
                    agent = Agent(self.learner)
                    environment = Environment(X.float(),y.float())
                    
                    # max number of possible actions
                    max_actions = environment.get_number_of_actions()
                    
                    # all seeds in the set
                    tseeds = get_seeds(self.searchspace,train_set)
                    # get new seeds not in tseeds
                    while(True):
                        newseeds = torch.randint(1000,(5,))
                        duplicate = sum([str(newseed) in tseeds for newseed in newseeds])
                        if duplicate<1:
                            break
                    del tseeds
                    
                    # evaluate performance for every seed
                    print("Starting rollout...",flush=True)
                    trajectory,states,kernel_state_dicts = agent.rollout(environment,100,newseeds,device=device) # return traj per seed to make it comparable
                    with torch.no_grad():
                        # critizise each action with Experts, by adding them to the new dataset
                        expert_actions = environment.expert_actions(trajectory,states,kernel_state_dicts,newseeds)
                        
                    print("Creating state actions from expert actions...",flush=True)
                    for t in range(len(expert_actions.keys())):
                        # Append new state-action-pairs to global dict
                        state_action_pairs_dagger[t] = (torch.cat((state_action_pairs_dagger[t][0], states[t].repeat_interleave(expert_actions[t].shape[1],dim=0).cpu() ))
                                                        ,torch.cat((state_action_pairs_dagger[t][1] ,expert_actions[t].view(-1,expert_actions[t].shape[2]) )) )
                        
                    # deleting old variables and free memory
                    del expert_actions,trajectory,states,kernel_state_dicts
                    torch.cuda.empty_cache()
                    gc.collect()                  
                        
                # Create new dataset and sampler for this training run
                self.train = History_Increasing_Dataset(state_action_pairs_dagger)
                self.history_sampler = torch.utils.data.RandomSampler(state_action_pairs_dagger.keys())
                # every epoch train with DAgger states, and every 5 epochs test with rollout
                print("Starting imitation learning...",flush=True)
                if((dagger_epoch+1) % 5) == 0:
                    self.imitation_learning(epochs+(dagger_epoch*50), epochs+((dagger_epoch+1)*50), 50,do_rollout=True,verbose=False) # do Rollout
                else:
                    self.imitation_learning(epochs+(dagger_epoch*50), epochs+((dagger_epoch+1)*50), 50,do_rollout=False,verbose=False) # do not Rollout
                
                # checkpoint state action pairs
                with open("../models/checkpoints/{}{}/state_action_pairs{}.pkl".format(self.searchspace,self.postfix,dagger_epoch),"wb") as f:
                    pkl.dump(state_action_pairs_dagger,f)
                
        # Save model and test stats
        torch.save(self.learner.state_dict(),"../models/tests/learner_{}{}_test.pt".format(self.searchspace,self.postfix))
        
        with open("../teststatistics/tests_{}{}.pkl".format(self.searchspace,self.postfix),"wb") as f:
            pkl.dump({"average_normalized_regret_hist":self.average_normalized_regret_hist,"average_rank_hist":self.average_rank_hist,"traj_hist":self.traj_hist,"traj_accs_hist":self.traj_accs_hist,"time_to_best_hist":self.time_to_best_hist,"regret_hist":self.regret_hist,"regret_curve_hist":self.regret_curve_hist,"average_regret_hist":self.average_regret_hist,"testdatasets":self.testdatasets},f)
        
        # delete checkpoints
        # shutil.rmtree("../models/checkpoints/{}{}".format(self.searchspace,self.postfix))
        
        return self.traj_hist

    def imitation_learning(self,epochs_low,epochs,testinterval,do_rollout=True,verbose=True,device=torch.device("cuda")):
        # if no interval is given, just use (epochs / 10)
        if testinterval == 0:
            testinterval = int(epochs / 10)
        
        for e in range(epochs_low, epochs):
            if(verbose):
                print('.',end="",flush=True)
            # in every epoch, go over all history lengths
            e_loss = 0
            for i in self.history_sampler:
                
                if len(self.train.dataset[i][0]) == 0: # skip this history length if there are no entries
                    continue
                        
                self.train.sethistlength(i)
                loader = torch.utils.data.DataLoader(self.train, batch_size = self.batch_size, shuffle = True)
                                
                for b,data in enumerate(loader):
                    self.optimizer.zero_grad()
                    
                    # Perform forward pass
                    outputs = self.learner(data[0].float().cuda())
                    
                    # Compute loss
                    loss = self.loss_fn(outputs, data[1].cuda())
            
                    # Perform backward pass
                    loss.backward()
            
                    # Perform optimization
                    self.optimizer.step()
                    
                    e_loss += loss.item() / (len(loader)*len(self.history_sampler))
                
            # print loss
            if ((e+1) % 50) == 0:
                print("Epoch {} loss: {}".format(e+1,e_loss))
                sys.stdout.flush()
          
            # save checkpoint every 200 epochs
            if ((e+1) % testinterval) == 0:
                print("Saving checkpoint")
                if not os.path.isdir("../models/checkpoints/{}{}".format(self.searchspace,self.postfix)):
                    os.mkdir("../models/checkpoints/{}{}".format(self.searchspace,self.postfix))
                torch.save(self.learner.state_dict(),"../models/checkpoints/{}{}/{}.pt".format(self.searchspace,self.postfix,e+1))
                torch.save(self.optimizer.state_dict(),"../models/checkpoints/{}{}/{}_optim.pt".format(self.searchspace,self.postfix,e+1))
                        
            # Validate model with rollout
            if (((e+1) % testinterval)==0 and do_rollout):     
            
                # Loop over testdatasets
                for dset in self.testdatasets:
                    print("Starting to validate model on dataset {}".format(dset))           
                    # load data for testset
                    X,y = data_loader(self.searchspace,dset,self.hdlr,device=device)
                    
                    # Create Agent and Environment
                    agent = Agent(self.learner)
                    environment = Environment(X.float(),y.float())
                    
                    # max number of possible actions
                    max_actions = environment.get_number_of_actions()
                    
                    # all seeds in the set
                    tseeds = get_seeds(self.searchspace,dset)
                    
                    # evaluate performance for every seed
                    trajectory,_,_ = agent.rollout(environment,100,tseeds,device=device) # return traj per seed to make it comparable
                    
                    # compute statitics of rollout for every single seed
                    traj_accs =  y[trajectory].squeeze(-1)
                    regret,regret_curve = compute_regret(traj_accs,y)
                    time_to_best = torch.max(traj_accs,axis=1)[1] # indices of when the best acc was hit
                    average_regret = regret[:max_actions+10].mean() 
                    print("Average regret on dataset {}: {}".format(dset,average_regret))
                    
                                 
                    # save to history
                    self.traj_hist[dset] = torch.cat(( self.traj_hist[dset] , trajectory[:,:max_actions+10].unsqueeze(0).cpu())) 
                    self.traj_accs_hist[dset] = torch.cat(( self.traj_accs_hist[dset] , traj_accs[:,:max_actions+10].unsqueeze(0).cpu())) 
                    self.time_to_best_hist[dset] = torch.cat(( self.time_to_best_hist[dset] , time_to_best.unsqueeze(0).cpu())) 
                    self.regret_hist[dset] = torch.cat(( self.regret_hist[dset] , regret[:,:max_actions+10].unsqueeze(0).cpu())) 
                    self.regret_curve_hist[dset]= torch.cat(( self.regret_curve_hist[dset] , regret_curve.unsqueeze(0).cpu())) 
                    self.average_regret_hist[dset] = torch.cat(( self.average_regret_hist[dset] , average_regret.unsqueeze(0).cpu())) 
                    
                                    
                # compute statistics over all testdatasets & compare to baseline
                latest_regret_curves = torch.stack([v[-1] for k,v in self.regret_curve_hist.items()]).view(-1,regret_curve.shape[1])
                average_normalized_regret = torch.mean(latest_regret_curves,axis=0)
                average_rank = compute_ranks(self.baseline_normalized_regrets,latest_regret_curves)#.to(device=device)
                
                self.average_normalized_regret_hist = torch.cat((self.average_normalized_regret_hist,average_normalized_regret.unsqueeze(0)))
                self.average_rank_hist = torch.cat((self.average_rank_hist,average_rank.unsqueeze(0)))
                
                print("Average normalized regret: ",average_normalized_regret)
                print("Average rank: ",average_rank)
                
                # Checkpoint the stats
                if not os.path.isdir("../models/checkpoints/{}{}".format(self.searchspace,self.postfix)):
                    os.mkdir("../models/checkpoints/{}{}".format(self.searchspace,self.postfix))
                with open("../models/checkpoints/{}{}/{}_stats.pkl".format(self.searchspace,self.postfix,e+1),"wb") as f:
                     pkl.dump({"average_normalized_regret_hist":self.average_normalized_regret_hist,"average_rank_hist":self.average_rank_hist,"traj_hist":self.traj_hist,"traj_accs_hist":self.traj_accs_hist,"time_to_best_hist":self.time_to_best_hist,"regret_hist":self.regret_hist,"regret_curve_hist":self.regret_curve_hist,"average_regret_hist":self.average_regret_hist,"testdatasets":self.testdatasets},f)
       