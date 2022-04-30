# Learning to Optimize Hyperparameters (L2O-HP)

This repository contains the code to the master thesis "Learning to Optimize Hyperparameters with Guided Policy Search" by Simon Boeder, supervised by Hadi Samer Jomaa.

## Requirements
To run this project, you need the requirements from the `requirements.txt` file. 

## Generating Guiding Trajectories
In order to train the models, guiding trajectories generated from teacher policies (e.g. acquisition functions like EI, POI, UCB, MES) are required. You need the benchmark datasets downloaded in the root folder of the repository, and the HPO-B repository in the `src` directory. The benchmark datasets can be found under [[2](#f2)] and [[3](#f2)].

In order to generate new trajectories, you can run the `data_generation.py` file like the following:

```
python data_generator.py -d 'benchmarkname' -k 'dsetname1' 'dsetname2' 
```

- `-d` is the search space. Available are 'hpo_data', which will use the UCI benchmark search space, or one of the HPO-B search space ID's [[1](#f1)] (e.g. '5891').
- `-k` is a list of meta-dataset ID's of the selected search space.

## Training
To start training, use the `run.py` file. You need a model configuration in the `model_configs` folder that sets the hyperparameters of the model. See below for usage.

```
python run.py  -s '5889' -e 10000 -c 1000 -a 10 -f 0
```

- `-s` is the search space for which a model should be trained. The program will automatically fetch trajectories from the `trajectories` folder.
- `-e` is the number of epochs of regular training. The models from the thesis trained for 10,000 epochs.
- `-c` is the interval of epochs in which testing via roll out should be performed. As this takes potentially lots of time, consider not testing during training (e.g. setting this equal to the number of epochs).
- `-a` is the number of DAgger epochs to apply after the regular training is finished. Each DAgger epoch corresponds to rolling out 5 new trajectories per training dataset and training for 50 epochs. If left empty or 0, no DAgger retraining will be applied. The model will be saved as a vanilla L2O-HP model.
- `-f` is the ID of the model config that should be used. This number is the name of the config file, e.g. 0 = `0.ini` in the `model_configs` folder.
- `-t` are the test datasets that should be used. If left empty, automatically 30% of the available datasets will be held back for testing. The test datasets that were held back can be queried via the `Utility.get_testdatasets(searchspace)` method.

## Hyperparameter settings
The hyperparameters of the model are set in the `*.ini` files in the `model_configs` directory. Use the following:

- hidden_size = Latent space dimensionality of the Prediction Head.
- transformer_hidden_size = Latent space dimensionality of the Set Transformer.
- transformer_out_features = Output dimensionality of the Set Transformer.
- lr = Learning rate of the optimizer.
- n_hidden_layers = Number of hidden layers in the Prediction Head
- batch_size = Batch size of the data loader. How many state-action pairs are loaded at the same time.

There are more hyperparameters that are not yet changeable. Future updates will introduce more possibilities.

## Inference & Baselines
You can run inference with L2O-HP and the baselines with the `Baselines.py` file.

``` 
python Baselines.py  -s '5889' -a 'L2OHP' -t '14971' '3954'
```
- `-s` is the search space for which a baseline or model should be run. The program will automatically fetch the trained model from the `models` folder.
- `-a` is the algorithm for which inference should be run. Available are ["L2OHP", "L2OHPDAgger", "CMAES", "SMAC", "HEBO", "BOHAMIANN", "BO"]
- `-t` is a list of test datasets for which you want to run inference. If left empty, it will take the test datasets used during training, if any. 

The program will automatically fetch the corresponding L2O-HP (or DAgger) model for the search space.

## Links
<a name="f1">1</a>: https://arxiv.org/abs/2106.06257
<a name="f2">2</a>: https://github.com/releaunifreiburg/HPO-B
<a name="f3">3</a>: https://github.com/hadijomaa/dataset2vec