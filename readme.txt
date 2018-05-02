This code is implementation of branchLSTM model from the paper 
Turing at SemEval-2017 Task 8: Sequential Approach to Rumour Stance
Classification with Branch-LSTM
https://www.aclweb.org/anthology/S/S17/S17-2083.pdf

This code uses Python 2.7 with Lasagne and Theano libraries

Data is taken from and can be downloaded from RumourEval website:
http://alt.qcri.org/semeval2017/task8/index.php?id=data-and-tools

Additionally, GoogleNews pre-trained word vectors should be downloaded from:
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

This code contains the following files:

 - preprocssing.py
   Preprocesses the data into the format suitable for use with Lasagne model 
   and saves arrays into saved_data folder. 
   
   Contains the following functions: 
   
   load_dataset - loads data into python dictionary
   tree2branches - splits tree into branches
   cleantweet - replaces urls and pics in the tweet with 
                'picpicpic' and 'urlurlurl' tokens
   str_to_wordlist - applies cleantweet and tokenizes the tweet, 
                     optionally removes stopwords
   loadW2vModel - loads W2V model into global variable
   sumw2v - turns tweet into sum or average of it's word's vectors. 
   getW2vCosineSimilarity - computes cosine similarity between tweets
   tweet2features - extracts features from tweet.
   convertlabel - converts str labels to int
   
 - training.py
   Contains the following functions: 
   
   build_nn - defines architecture of Neural Network
   iterate_minibatches - splits training data into mini-batches and 
                         returns iterator
   objective_train_model - trains model on the training set, 
                           evaluates on development set and returns output 
                           in the format suitable for use with hyperopt package 
            
 - predict.py
   Contains the following functions: 
   
   eval_train_model - re-trains model on train+dev set and 
                      evaluates on test set
   
 - outer.py  - 
   Contains the following functions: 
   
   parameter search - defines parameter space,performs parameter search using 
                      objective_train_model and hyperopt TPE search.
   convertlabeltostr - converts int label to str
   eval - passes parameters to eval_train_model, does results postprocessing 
          to fit with scorer.py and saves results
   main - brings all together, controls command line arguments
   

To run the code:

0) Install prerequisites:

 - numpy
 - sklearn
 - nltk

 - gensim
 - theano 
 - lasagne
 - hyperopt

1) Download data and word vectors
2) Run preprocessing.py

python preprocessing.py

3) Run outer.py

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


The official evaluation scritp is scorer.py, outer.py saves predictions.txt 
that can be used with scorer.py


Feel free to email E.Kochkina@warwick.ac.uk if you have any questions. 


