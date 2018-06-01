#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Contains the following functions: 
   
parameter search - defines parameter space,performs parameter search using 
                      objective_train_model and hyperopt TPE search.
convertlabeltostr - converts int label to str
eval - passes parameters to eval_train_model, does results postprocessing 
          to fit with scorer.py and saves results
main - brings all together, controls command line arguments

Run outer.py

python outer.py

outer.py has the following options:

python outer.py --search=True --ntrials=10 --params="output/bestparams.txt"

--search  - boolean, controls whether parameter search should be performed
--ntrials - if --search is True then this controls how many different 
            parameter combinations should be assessed
--params - specifies filepath to file with parameters if --search is false

-h, --help - explains the command line 


If performing parameter search, then execution will take long time 
depending on number of trials, size and number of layers in parameter space. 
Use of GPU is highly recommended. 

If running with default parametes then search won't be performed and 
parameters will be used from 'output/bestparams.txt'

"""
import timeit
import os
import pickle
import json
os.environ["THEANO_FLAGS"]="floatX=float32"
#os.environ["THEANO_FLAGS"]="floatX=float32,dnn.enabled=False,cxx=icpc,
#                            device=gpu0,nvcc.compiler_bindir=icpc,
#                            gcc.cxxflags=-march=native"
from hyperopt import fmin, tpe, hp, Trials #rand
from training import objective_train_model
from predict import eval_train_model
from optparse import OptionParser
#%%


def parameter_search(ntrials):
    start = timeit.default_timer()
    trials = Trials()
   
    search_space= { 'num_dense_layers': hp.choice('nlayers',[1,2,3,4]),
                    'num_dense_units': hp.choice('num_dense', [100, 200, 300,
                                                               400, 500]), 
                    'num_epochs': hp.choice('num_epochs',  [30, 50, 70, 100]),
                    'num_lstm_units': hp.choice('num_lstm_units',  [100, 200,
                                                                    300]),
                    'num_lstm_layers': hp.choice('num_lstm_layers', [1,2]),
                    'learn_rate': hp.choice('learn_rate',[1e-4, 3e-4, 1e-3]), 
                    'mb_size': hp.choice('mb_size', [32, 64, 100, 120]),
                    'l2reg': hp.choice('l2reg', [0.0, 1e-4, 3e-4, 1e-3]),
                    'rng_seed': hp.choice('rng_seed', [364]) 
    }
       
    best = fmin(objective_train_model,
        space=search_space,
        algo=tpe.suggest,
        max_evals=ntrials,
        trials=trials)
        
    print ("Best params: ", best)
    
    params = trials.best_trial['result']['Params']
        
    out_path = 'output'
    if not os.path.exists(out_path):
            os.makedirs(out_path)
    
    f = open(os.path.join(out_path,'trials.txt'), "w+")
    pickle.dump(trials, f)
    f.close()
    
    f = open(os.path.join(out_path,'bestparams.txt'), "w+")
    pickle.dump(params, f)
    f.close()
    
    print ("saved trials and params")
    
    stop = timeit.default_timer()
    print ("Time: ",stop - start)  
    
    return params

#%%


def convertlabeltostr(label):
    if label==0:
        return("support")
    elif label==1:
        return("comment")
    elif label==2:
        return("deny")
    elif label==3:
        return("query")
    else:
        print(label)  
        
#%%


def eval(params):
    start = timeit.default_timer()
    result = eval_train_model(params)
    # Convert result to scorer.py format
    keys = pickle.loads(result['attachments']['ID'])
    values = pickle.loads(result['attachments']['Predictions'])
    values = [convertlabeltostr(s) for s in values] 
    result_dictionary = dict(zip(keys, values))
    
    out_path = 'output'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    f = open(os.path.join(out_path,'result.txt'), "w+")
    pickle.dump(result, f)
    f.close()

    with open(os.path.join(out_path,'predictions.txt'), 'w+') as outfile:
        json.dump(result_dictionary, outfile)
    print ("saved result and predictions")
    stop = timeit.default_timer()
    print ("Time: ",stop - start)
    
#%%


def main():
    parser = OptionParser()
    parser.add_option(
            '--search', dest='psearch', default=False,
            help='Whether parameter search should be done: default=%default')
    parser.add_option('--ntrials', dest='ntrials', default=10,
                      help='Number of trials: default=%default')
    parser.add_option(
            '--params', dest='params_file', default='output/bestparams.txt',
            help='Location of parameter file: default=%default')
    
    
    (options, args) = parser.parse_args()
    psearch = options.psearch
    ntrials = int(options.ntrials)
    params_file = options.params_file
    
    if psearch:
        print "\nStarting parameter search...\n"
        params = parameter_search(ntrials)
        print(params)
        eval(params)
    else:
        with open(params_file, 'rb') as f:
            print "\nLoading best set of model parameters...\n"
            params = pickle.load(f)
        print (params)
        eval(params)
        
#%%


if __name__ == '__main__':
    main()
