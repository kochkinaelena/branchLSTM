# branchLSTM

This code is the implementation of the branchLSTM model from the paper
*Turing at SemEval-2017 Task 8: Sequential Approach to Rumour Stance
Classification with Branch-LSTM*, available [here](
https://www.aclweb.org/anthology/S/S17/S17-2083.pdf).

This version of the code uses Python 2.7 with the Lasagne and Theano libraries.

## Installation and usage instructions

Start by cloning this repository, and then follow the instructions below to download the additional datasets and set up the dependencies.


### 1. Download datasets

The datasets from the SemEval-2017 Task 8 challenge and a Word2Vec model pretrained on the Google News dataset are required.

These files should be placed in the `downloaded_data` folder.
Instructions for acquiring these files may be found in the [README](downloaded_data/README.md) inside the `downloaded_data` folder.

### 2a. Installation: local machine

We recommend creating a new virtual environment and installing the required packages via `pip`.
```
cd <your-branchLSTM-directory>
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

### 2b. Installation: Microsoft Azure Data Science VM

While it is possible to load and apply the final model on a typical desktop/laptop, GPU resources are highly recommended if you want to run the full parameter search.

The instructions below guide you though setting up branchLSTM on a Microsoft Azure Virtual Machine.
Free trial accounts are available for [students](https://azure.microsoft.com/en-gb/free/students/) and [other users](https://azure.microsoft.com/en-gb/offers/ms-azr-0044p/).

Running the parameter search should take approximately (***edit***) on an NC6 VM.

#### Set up the VM

We will use Microsoft's *Data Science Virtual Machine for Linux (Ubuntu)*.

This VM comes with CUDA and associated tools pre-installed.
If preferred, you can use a VM that is not preconfigured (for instance, the *Ubuntu Server 16.04 LTS*), but you will have to install CUDA to use the GPU-enabled features.

Instructions for creating the VM are available [here](https://docs.microsoft.com/en-gb/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro).
When configuring the VM, select the following options:
* Basics/VM disk type: HDD
* Size: NC6 (if not available, change the location and ensure that disk type is HDD)

#### Install required packages

* Create a virtual environment for this installation.
The *Data Science Virtual Machine for Linux (Ubuntu)* comes with Anaconda Python, so we use `conda` to create the virtual environment.
```
cd <your-branchLSTM-directory>
conda create -n env python=2.7
source activate env
```
* Install the packages from `requirements.txt`.
```
pip install -r requirements.txt
```
* Additionally, install the `pygpu` package (we use `conda` as it can also install `libgpuarray`).
```
conda install pygpu=0.6.9
```
* Add the following to your `.bashrc` file:  
`export CUDA_HOME=/usr/local/cuda-8.0`  
`export LD_LIBRARY_PATH=${CUDA_HOME}/lib64`  
`export PATH=${CUDA_HOME}/bin:${PATH}`  
and then reload.
```
source ~/.bashrc
```

### 3. Construct and apply the model

* Run the preprocessing stage to convert the data into a format that is compatible with Lasagne.
```
python preprocessing.py
```
* [Optional; GPU recommended] Determine the optimal set of hyperparameters, which will be saved to `output/bestparams.txt`. If GPU resources are unavailable, skip this step and use the hyperparameters saved in `output/bestparams_semeval2017.txt`.
```
THEANO_FLAGS='floatX=float32,device=cuda' python outer.py --search=True --ntrials=100
```
* Construct the model using the optimal set of hyperparameters and apply to the test dataset.
```
THEANO_FLAGS='floatX=float32' python outer.py
```
By default, the command above reads the hyperparameters from `output/bestparams_semeval2017.txt`.
Hyperparameters saved elsewhere can be specified with
```
  THEANO_FLAGS='floatX=float32' python outer.py --params='output/bestparams.txt'
```
The results are saved in `output/predictions.txt` in a format compatible with the scoring script.
* Evaluate the performance of the model with the official SemEval-2017 scoring script (this script uses Python 3 rather than Python 2, so we specify the correct Python version).
```
python3 scorer/scorerA.py "subtaska.json" "output/predictions.txt"
```


## Code structure

`preprocessing.py`
  * Preprocesses the data into format suitable for use with Lasagne and saves arrays into the `saved_data` folder
  * `load_dataset` loads data into a python dictionary
  * `tree2branches` splits tree into branches
  * `cleantweet` replaces urls and pictures in the tweet with `picpicpic` and `urlurlurl` tokens
  * `str_to_wordlist` applies cleantweet and tokenizes the tweet, optionally removing stopwords
  * `loadW2vModel` loads W2V model into global variable
  * `sumw2v` turns tweet into sum or average of its words' vectors
  * `getW2vCosineSimilarity` computes cosine similarity between tweets
  * `tweet2features` extracts features from tweet
  * `convertlabel` converts `str` labels to `int`

`training.py`
  * `build_nn` defines the architecture of the Neural Network
  * `iterate_minibatches` splits the training data into mini-batches and returns iterator
  * `objective_train_model` trains the model on the training set, evaluates on development set and returns output in a format suitable for use with the `hyperopt` package

`predict.py`
  * `eval_train_model` re-trains the model on training and development set and evaluates on the test set

`outer.py`
  * `parameter_search` defines parameter space, performs parameter search using `objective_train_model` and `hyperopt` TPE search.
  * `convertlabeltostr` converts `int` labels to `str`
  * `eval` passes parameters to `eval_train_model`, does results postprocessing to fit with `scorer.py` and saves results
  * `main` brings all together, controls command line arguments
  * The following options are available:
    * `--search`: boolean, controls whether parameter search should be performed (default `--search=False`)
    * `--ntrials`: if `--search=True` then this controls how many different parameter combinations should be assessed (default `--ntrials=10`)
    * `--test`: boolean, if `--search=True`, this sets the type of parameter search to be run (default `--test=False`;)
    * `--params`: specifies filepath to file with parameters if `--search=False` (default `--params=output/bestparams_semeval2017`)
    * `-h`, `--help`: explains the command line arguments


`bestparams_semeval2017.txt`
  * This file stores the parameters used in the competition and paper

## Contact

Feel free to email E.Kochkina@warwick.ac.uk if you have any questions.
