# branchLSTM

This code is the implementation of the branchLSTM model from the paper
*Turing at SemEval-2017 Task 8: Sequential Approach to Rumour Stance
Classification with Branch-LSTM*, available [here](
https://www.aclweb.org/anthology/S/S17/S17-2083.pdf).

This version of the code uses Python 2.7 with the Lasagne and Theano libraries.

## Option 1: Installation on a local machine

To begin, clone this repository.
```
git clone https://github.com/kochkinaelena/branchLSTM.git
```

The datasets from the SemEval-2017 Task 8 challenge and a Word2Vec model pretrained on the Google News dataset are required.

These files should be placed in the `downloaded_data` folder.
Instructions for acquiring these files may be found in the [README](downloaded_data/README.md) inside the `downloaded_data` folder.

We recommend creating a new virtual environment and installing the required packages via `pip`.
```
cd <your-branchLSTM-directory>
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```


## Option 2: Installation on a Microsoft Azure VM

While it is possible to load and apply the final model on a typical desktop/laptop, GPU resources are highly recommended if you want to run the full parameter search.

The instructions below guide you though setting up branchLSTM on a Microsoft Azure Virtual Machine.
Free trial accounts are available for [students](https://azure.microsoft.com/en-gb/free/students/) and [other users](https://azure.microsoft.com/en-gb/offers/ms-azr-0044p/).

Running the parameter search should take approximately (***edit***) on an NC6 VM.

### Set up the VM

Once you have your account, log into the Azure portal and start the process of creating your VM.
* Click on "Create a resource" and select "Ubuntu Server 16.04 LTS".
* In the "Basics" panel, you will need to select "VM disk type = HDD". Other options may be set as you wish (see [this page](https://docs.microsoft.com/en-us/azure/virtual-machines/linux/quick-create-portal) for general instructions).
* In the "Size" panel, select NC6 - this is the smallest GPU available, but is sufficient for our purposes.
* Change the options in the final panels if you want, and then create the resource.

_Note: If you have trouble finding the NC6 option, make sure the HDD disk type is specified or try changing the location._

### Install CUDA

Once you have logged into the VM, run the commands below to install the CUDA toolkit.
```
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
mv cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```

Specify the paths to CUDA by adding the following to your `.bashrc` file:  
```
export CUDA_HOME=/usr/local/cuda-8.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
export PATH=${CUDA_HOME}/bin:${PATH}
```
and then reload with
```
source ~/.bashrc
```

### Download branchLSTM

Clone the git repo and move into the branchLSTM directory.
```
git clone https://github.com/kochkinaelena/branchLSTM.git
cd branchLSTM
```
Follow the instructions in the [README](../downloaded_data/README.md) for details of how to download the datasets needed for this project.

### Install python packages

`pip` is not preinstalled on this VM, so we must do that before creating a virtual environment.
Having activated the virtual environment, we install all required packages listed in `requirements.txt`.
```
sudo apt install python-pip
pip install virtualenv
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```


## Construct and apply the model

* Run the preprocessing stage to convert the data into a format that is compatible with Lasagne.
```
python preprocessing.py
```
* _[Optional; GPU recommended]_ Determine the optimal set of hyperparameters, which will be saved to `output/bestparams.txt`. If GPU resources are unavailable, skip this step and use the hyperparameters saved in `output/bestparams_semeval2017.txt`.
```
THEANO_FLAGS='floatX=float32,device=gpu' python outer.py --search=True --ntrials=100
```
* Construct the model using the optimal set of hyperparameters and apply to the test dataset.
```
THEANO_FLAGS='floatX=float32' python outer.py
```
By default, the command above reads the hyperparameters from `output/bestparams_semeval2017.txt`.
Hyperparameters saved elsewhere can be specified with (for example)
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
