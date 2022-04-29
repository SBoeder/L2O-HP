# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:55:18 2022

@author: simon

Hpyerparameter Optimization script
"""
import torch
from Trainer import Trainer
import argparse
import time
# import numpy as np
# import cProfile,pstats
# import io
import os
import pickle as pkl
#%%

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s',"--searchspace")
parser.add_argument('-d',"--datasets",nargs="*",default="0",type=str) # leave empty to use all datasets
parser.add_argument('-e',"--epochs",default=200,type=int) # number of epochs to train each model
args = parser.parse_args()
print("arguments read...",flush=True)
print("Searchspace: "+args.searchspace,flush=True)


#%% 
def run_hpo():
        
    results = []
    # read search grid
    with open("hpoconf.pkl","rb") as f:
        config = pkl.load(f)
    
    # load checkpoint if available, skip the ones already done
    doneconfs = []
    if os.path.exists(path+"/{}_hpo_results.pkl".format(args.searchspace)):
        with open(path+"/{}_hpo_results.pkl".format(args.searchspace),"rb") as f:
            results = pkl.load(f)
        
        for r in results:
           doneconfs.append(r["config"])
        
    # run training function
    for i,run in enumerate(config):    
        print("STARTING RUN {}".format(i))
        if run in doneconfs:
            print("Already done, loaded from checkpoint.",flush=True)
            continue
                    
        start = time.time() # track start time of run
        trainer = Trainer(args.searchspace,args.datasets,
                  hidden_size=run["hidden-size"],
                  transformer_hidden_size=run["transformer-hidden-size"],
                  transformer_out_features=run["hidden-size"],
                  n_hidden_layers=run["n-hidden-layers"],
                  heads=4,
                  lr=run["learning-rate"],
                  seed=42,
                  batch_size=128)
        temp_result = trainer.train_learner_folds(args.epochs,folds=5,device=torch.device("cuda"))
        
        end = time.time() # end time of run
        
        # report average test results to tune
        avg_test_loss = 0
        avg_train_loss = 0
        for fold,res in temp_result.items():
            testloss = res[1][-1].item()
            trainloss = res[0][-1].item()
            avg_test_loss += testloss / len(temp_result)
            avg_train_loss += trainloss / len(temp_result)
      
        results.append({"config":run,"results":temp_result,"avg_train_loss":avg_train_loss,"avg_test_loss":avg_test_loss,"runtime":round(end-start,2)})
            
        # checkpoint results
        with open(path+"/{}_hpo_results.pkl".format(args.searchspace),"wb") as f:
            pkl.dump(results,f)
  
    return results


#%%
path = "../models/hpo"
# Run experiment
foldresults = run_hpo()
with open(path+"/{}_hpo_results.pkl".format(args.searchspace),"wb") as f:
    pkl.dump(foldresults,f)

#%% Profiling section
# profiler = cProfile.Profile()
# profiler.enable()
# foldresults = run_hpo(config)
# profiler.disable()
# s = io.StringIO()
# stats = pstats.Stats(profiler,stream = s).sort_stats('tottime')
# stats.strip_dirs()
# stats.print_stats()   
# with open('stats.txt', 'w+') as f:
#     f.write(s.getvalue())
