"""
This is outer preprocessing file

To run:
    
python prep_pipeline.py

Main function has parameters that can be changed:
    
dataset ('RumEv' or 'fullPHEME') 
and feats ('text' or 'SemEval')

"""
from read_fullPHEME_data import read_fullPHEME
from read_RumEv_data import read_RumEv
from transform_feature_dict import transform_feature_dict
from extract_thread_features import extract_thread_features_incl_response
import help_prep_functions
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
#%%


def convert_label(label):
    if label == "true":
        return(0)
    elif label == "false":
        return(1)
    elif label == "unverified":
        return(2)
    else:
        print(label)


def prep_pipeline(dataset='RumEv', feature_set=['avgw2v']):
    path = 'saved_data'+dataset
    folds = {}
    if dataset == 'RumEv':
        folds = read_RumEv()
    else:
        folds = read_fullPHEME()
    help_prep_functions.loadW2vModel()
    for fold in folds.keys():
        print(fold)
        feature_fold = []
        tweet_ids = []
        fold_stance_labels = []
        labels = []
        ids = []
        for conversation in folds[fold]:
            thread_feature_dict = extract_thread_features_incl_response(
                                  conversation)

            thread_features_array, thread_stance_labels, branches = transform_feature_dict(
                                   thread_feature_dict, conversation,
                                   feature_set=feature_set)
            
            fold_stance_labels.extend(thread_stance_labels)
            tweet_ids.extend(branches)
            feature_fold.extend(thread_features_array)
            for i in range(len(thread_features_array)):
                labels.append(convert_label(conversation['veracity']))
                ids.append(conversation['id'])
        feature_fold = pad_sequences(feature_fold, maxlen=None,
                                     dtype='float32',
                                     padding='post',
                                     truncating='post', value=0.)

        fold_stance_labels = pad_sequences(fold_stance_labels, maxlen=None,
                                           dtype='float32',
                                           padding='post', truncating='post',
                                           value=0.)
#
        tweet_ids = pad_sequences(tweet_ids, maxlen=None, dtype='float32',
                                  padding='post', truncating='post',
                                  value=0)
        labels = np.asarray(labels)
        path_fold = os.path.join(path, fold)
        if not os.path.exists(path_fold):
            os.makedirs(path_fold)

        np.save(os.path.join(path_fold, 'train_array'), feature_fold)
        np.save(os.path.join(path_fold, 'labels'), labels)
        np.save(os.path.join(path_fold, 'fold_stance_labels'),
                fold_stance_labels)
        np.save(os.path.join(path_fold, 'ids'), ids)
#        np.save(os.path.join(path_fold,'tweet_ids'),char_tweet_ids) 
        np.save(os.path.join(path_fold, 'tweet_ids'), tweet_ids)
        
def main(data ='RumEv', feats = 'text'):
    if data == 'RumEv':
        if feats == 'text':
            prep_pipeline(dataset='RumEv', feature_set=['avgw2v'])
        elif feats == 'SemEvalfeatures':
            SemEvalfeatures = ['avgw2v', 'hasnegation', 'hasswearwords',
                               'capitalratio', 'hasperiod', 'hasqmark',
                               'hasemark', 'hasurl', 'haspic',
                               'charcount', 'wordcount', 'issource',
                               'Word2VecSimilarityWrtOther',
                               'Word2VecSimilarityWrtSource',
                               'Word2VecSimilarityWrtPrev']
            prep_pipeline(dataset='RumEv', feature_set=SemEvalfeatures)

    elif data == 'fullPHEME':
        if feats == 'text':
            prep_pipeline(dataset='fullPHEME', feature_set=['avgw2v'])
        elif feats == 'SemEval':
            SemEvalfeatures = ['avgw2v', 'hasnegation', 'hasswearwords',
                               'capitalratio', 'hasperiod', 'hasqmark',
                               'hasemark', 'hasurl', 'haspic',
                               'charcount', 'wordcount', 'issource',
                               'Word2VecSimilarityWrtOther',
                               'Word2VecSimilarityWrtSource',
                               'Word2VecSimilarityWrtPrev']
            prep_pipeline(dataset='fullPHEME', feature_set=SemEvalfeatures)


if __name__ == '__main__':
    main()