# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 10:24:21 2021

@author: simon

Main entry file for running experiments / training 
"""

import argparse
from Trainer import Trainer
import torch
import time
import configparser
import warnings
warnings.filterwarnings("ignore")

# Track start time
start = time.time()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s',"--searchspace")
parser.add_argument('-d',"--datasets",nargs="*",default="0",type=str) # leave empty to use all datasets
parser.add_argument('-e',"--epochs",nargs="?",default=200,type=int)
parser.add_argument("-t", "--tests",nargs="*",default=0,type=str) # testdatasets to use for testing
parser.add_argument("-c", "--testtime",nargs="?",default=200,type=int) # training epochs between test runs
parser.add_argument("-f","--config",default="0",nargs="?",type=str) # which config file for model hyperparameters to use
parser.add_argument("-a","--dagger",default=0,nargs="?",type=int) # which config file for model hyperparameters to use
args = parser.parse_args()
print("arguments read...",flush=True)
    
# read the config file for hyperparameters
config = configparser.ConfigParser()
config.read("../model_configs/{}.ini".format(args.config))
# Run test trainer
trainer = Trainer(args.searchspace,args.datasets,
                  hidden_size=config.getint("DEFAULT","hidden_size"),
                  transformer_hidden_size=config.getint("DEFAULT","transformer_hidden_size"),
                  transformer_out_features=config.getint("DEFAULT","transformer_out_features"),
                  n_hidden_layers=config.getint("DEFAULT","n_hidden_layers"),
                  heads=4,
                  lr=config.getfloat("DEFAULT","lr"),
                  seed=42,
                  batch_size=config.getint("DEFAULT","batch_size"))

trainer.train_learner_rollout(args.epochs,args.testtime,args.tests,dagger_epochs=args.dagger,do_rollout=True, device=torch.device("cuda"))
    
# End time
end = time.time()

print("Execution finished after {} seconds".format(end-start))