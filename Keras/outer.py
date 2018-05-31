"""
Run outer.py
python outer.py
outer.py has the following options:
python outer.py --search=True --ntrials=10 --params="output/bestparams.txt"
--search  - boolean, controls whether parameter search should be performed
--ntrials - if --search is True then this controls how many different 
            parameter combinations should be assessed
--params - specifies filepath to file with parameters if --search is false
--task - which task to train, stance or veracity
--dataset - which dataset to use (only for veracity task), 
RumEval(train, dev, test) or fullPHEME(cross-validation)

-h, --help - explains the command line 
If performing parameter search, then execution will take long time 
depending on number of trials, size and number of layers in parameter space. 
Use of GPU is highly recommended. 
If running with default parametes then search won't be performed and 
parameters will be used from 'output/bestparams.txt'
"""
import pickle
#os.environ["THEANO_FLAGS"]="floatX=float32"
from optparse import OptionParser
from parameter_search import parameter_search
from evaluation_functions import eval_stance_LSTM_RumEv 
from evaluation_functions import eval_veracity_LSTM_RumEv
from evaluation_functions import eval_veracity_LSTM_CV
from objective_functions import objective_function_stance_branchLSTM_RumEv
from objective_functions import objective_function_veracity_branchLSTM_RumEv
from objective_functions import objective_function_veracity_branchLSTM_fullPHEME
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
    parser.add_option(
            '--task', dest='task', default='stance',
            help='Which task, stance or veracity: default=%default')
    parser.add_option(
            '--dataset', dest='data', default='RumEval',
            help='Which dataset to use (only for veracity task), RumEval(train, dev, test) or fullPHEME(cross-validation): default=%default')
    (options, args) = parser.parse_args()
    psearch = options.psearch
    ntrials = int(options.ntrials)
    params_file = options.params_file
    data = options.data
    task = options.task

    if task == 'stance':
        print ('Rumour Stance classification')
        
        if psearch:
            params = parameter_search(ntrials,
                             objective_function_stance_branchLSTM_RumEv, task)
        else:
            with open(params_file, 'rb') as f:
                params = pickle.load(f)
        print (params)
        eval_stance_LSTM_RumEv(params)

    elif task == 'veracity':
       print ('Rumour Veracity classification') 
       
       if data == 'RumEval':
           print ('Data: RumEval') 
           if psearch:
                params = parameter_search(ntrials,
                                 objective_function_veracity_branchLSTM_RumEv,
                                 task)
           else:
                with open(params_file, 'rb') as f:
                    params = pickle.load(f)
           print (params)
           eval_veracity_LSTM_RumEv(params)
            
       elif data == 'fullPHEME':
           print ('Data: fullPHEME') 
           if psearch:
                params = parameter_search(ntrials,
                             objective_function_veracity_branchLSTM_fullPHEME,
                             task)
           else:
                with open(params_file, 'rb') as f:
                    params = pickle.load(f)
           print (params)
           eval_veracity_LSTM_CV(params)

    else:
       print ('Task variable should be either stance or veracity') 
#%%


if __name__ == '__main__':
    main()    