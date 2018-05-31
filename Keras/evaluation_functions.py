"""
This code contains several versions of evaluation functions
"""
import numpy as np
from LSTM_models import LSTM_model_stance,LSTM_model_veracity
from LSTM_models import build_LSTM_model_veracity
from rmse import rmse
import os
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from keras.utils.np_utils import to_categorical
from branch2treelabels import branch2treelabels
import pickle
from copy import deepcopy
#%%


def eval_stance_LSTM_RumEv(params, branch=True):
    path = "preprocessing/saved_dataRumEv"
    x_train = np.load(os.path.join(path, 'train/train_array.npy'))
    y_train = np.load(os.path.join(path, 'train/fold_stance_labels.npy'))
    x_dev = np.load(os.path.join(path, 'dev/train_array.npy'))
    y_dev = np.load(os.path.join(path, 'dev/fold_stance_labels.npy'))
    x_test = np.load(os.path.join(path, 'test/train_array.npy'))
    y_test = np.load(os.path.join(path, 'test/fold_stance_labels.npy'))
    ids_test = np.load(os.path.join(path, 'test/tweet_ids.npy'))
    # join dev and train
    x_dev = pad_sequences(x_dev, maxlen=len(x_train[0]), dtype='float32',
                          padding='post', truncating='post', value=0.)
    y_dev = pad_sequences(y_dev, maxlen=len(y_train[0]), dtype='float32',
                          padding='post', truncating='post', value=0.)
    x_train = np.concatenate((x_train, x_dev), axis=0)
    y_train = np.concatenate((y_train, y_dev), axis=0)
    y_train_cat = []
    for i in range(len(y_train)):
        y_train_cat.append(to_categorical(y_train[i], num_classes=4))
    y_train_cat = np.asarray(y_train_cat)
    y_pred, confidence = LSTM_model_stance(x_train, y_train_cat,
                                           x_test, params)
    # get tree labels
    fids_test = ids_test.flatten()
    fy_pred = y_pred.flatten()
    fy_test = y_test.flatten()
    fconfidence = confidence.flatten()
    uniqtwid, uindices2 = np.unique(fids_test, return_index=True)
    uniqtwid = uniqtwid.tolist()
    uindices2 = uindices2.tolist()
    del uniqtwid[0]
    del uindices2[0]
    uniq_dev_prediction = [fy_pred[i] for i in uindices2]
    uniq_dev_label = [fy_test[i] for i in uindices2]
    uniq_dev_confidence = [fconfidence[i] for i in uindices2]
    mactest_P, mactest_R, mactest_F, _ = precision_recall_fscore_support(
                                            uniq_dev_label,
                                            uniq_dev_prediction,
                                            average='macro')
    mictest_P, mictest_R, mictest_F, _ = precision_recall_fscore_support(
                                            uniq_dev_label,
                                            uniq_dev_prediction,
                                            average='micro')
    test_P, test_R, test_F, _ = precision_recall_fscore_support(
                                    uniq_dev_label,
                                    uniq_dev_prediction)
    acc = accuracy_score(uniq_dev_label, uniq_dev_prediction)
    # evalute on test
    output = {
                  'Params': params,
                  'accuracy': acc,
                  'Macro': {'Macro_Precision': mactest_P,
                            'Macro_Recall': mactest_R,
                            'macro_F_score': mactest_F},
                  'Micro': {'Micro_Precision': mictest_P,
                            'Micro_Recall': mictest_R,
                            'micro_F_score': mictest_F},
                  'Per_class': {'Pclass_Precision': test_P,
                                'Pclass_Recall': test_R,
                                'Pclass_F_score': test_F},
                  'attachments': {'ID': uniqtwid, 'Label': uniq_dev_label,
                                  'Prediction': uniq_dev_prediction,
                                  'Confidence': uniq_dev_confidence}
                   }
    with open('output/Output_Stance_RumEval.pkl', 'wb') as outfile:
        pickle.dump(output, outfile)
    return output

#%%


def eval_veracity_LSTM_RumEv(params):

    path = 'preprocessing/saved_dataRumEv'
    x_train = np.load(os.path.join(path, 'train/train_array.npy'))
    y_train = np.load(os.path.join(path, 'train/labels.npy'))
    x_dev = np.load(os.path.join(path, 'dev/train_array.npy'))
    y_dev = np.load(os.path.join(path, 'dev/labels.npy'))
    x_test = np.load(os.path.join(path, 'test/train_array.npy'))
    y_test = np.load(os.path.join(path, 'test/labels.npy'))
    ids_test = np.load(os.path.join(path, 'test/ids.npy'))
    # join dev and train
    x_dev = pad_sequences(x_dev, maxlen=len(x_train[0]), dtype='float32',
                          padding='post', truncating='post', value=0.)
    x_train = np.concatenate((x_train, x_dev), axis=0)
    y_train = np.concatenate((y_train, y_dev), axis=0)
    y_train = to_categorical(y_train, num_classes=None)
    y_pred, confidence = LSTM_model_veracity(x_train, y_train, x_test, params)
    # get tree labels
    
    trees, tree_prediction, tree_label, tree_confidence = branch2treelabels(
                                                                ids_test,
                                                                y_test,
                                                                y_pred,
                                                                confidence)
    mactest_F = f1_score(tree_label, tree_prediction, average='macro')
    acc = accuracy_score(tree_label, tree_prediction)
    
    output = {'Params': params,
              'accuracy': acc,
              'macro_F_score': mactest_F,
              'ID':trees,
              'Label':tree_label,
              'Prediction':tree_prediction,
              'Confidence':tree_confidence
              }
    
    with open('output/Output_Veracity_RumEval.pkl', 'wb') as outfile:
        pickle.dump(output, outfile)
        
    return output

#%%


def eval_veracity_LSTM_CV(params):
    path = 'preprocessing/saved_datafullPHEME'
    folds = ['ebola-essien',
             'ferguson',
             'gurlitt',
             'ottawashooting',
             'prince-toronto',
             'putinmissing',
             'sydneysiege',
             'charliehebdo',
             'germanwings-crash']
#    max_branch_len = 25
    num_epochs = params['num_epochs']
    mb_size = params['mb_size']
    cv_prediction = []
    cv_label = []
    cv_confidence = []
    cv_ids = []
    allfolds = []
    for number in range(len(folds)):
        
        print(number)
        test = folds[number]
        train = deepcopy(folds)
        del train[number]
        
        x_train = np.load(os.path.join(path, 'ebola-essien', 'train_array.npy'))
        num_features = x_train.shape[2]
        
        model = build_LSTM_model_veracity(params,num_features)
        
        for t in train:
            x_train = np.load(os.path.join(path, t, 'train_array.npy'))
            y_train = np.load(os.path.join(path, t, 'labels.npy'))
            y_train = to_categorical(y_train, num_classes=3)
            model.fit(x_train, y_train, batch_size=mb_size,
                      epochs=num_epochs, shuffle=True, class_weight=None)
        # then eval on testing for this fold
        x_test = np.load(os.path.join(path, test, 'train_array.npy'))
        y_test = np.load(os.path.join(path, test, 'labels.npy'))
        ids_test = np.load(os.path.join(path, test, 'ids.npy'))
        pred_probabilities = model.predict(x_test, batch_size=mb_size)
        confidence = np.max(pred_probabilities, axis=1)
        y_pred = model.predict_classes(x_test, batch_size=mb_size)
       
        trees, tree_prediction, tree_label, tree_confidence = branch2treelabels(
                                                              ids_test,
                                                              y_test,
                                                              y_pred,
                                                              confidence)
        mactest_F = f1_score(tree_label, tree_prediction, average='macro')
        accuracy = accuracy_score(tree_label, tree_prediction)
        rmse_score = rmse(tree_label, tree_prediction, tree_confidence)
        perfold_result = {'fold': test,
                          'ID': trees,
                          'Label': tree_label,
                          'Prediciton': tree_prediction,
                          'Confidence': tree_confidence,
                          'macroF': mactest_F,
                          'accuracy': accuracy,
                          'rmse': rmse_score
                          }
        cv_ids.extend(trees)
        cv_prediction.extend(tree_prediction)
        cv_label.extend(tree_label)
        cv_confidence.extend(tree_confidence)
        allfolds.append(perfold_result)
    cv_label = np.asarray(cv_label)
    cv_prediction = np.asarray(cv_prediction)
    cv_confidence = np.asarray(cv_confidence)
    mactest_P, mactest_R, mactest_F, _ = precision_recall_fscore_support(
                                            cv_label,
                                            cv_prediction,
                                            average='macro')
    mictest_P, mictest_R, mictest_F, _ = precision_recall_fscore_support(
                                            cv_label,
                                            cv_prediction,
                                            average='micro')
    test_P, test_R, test_F, _ = precision_recall_fscore_support(cv_label,
                                                                cv_prediction)
    cvrmse_score = rmse(cv_label, cv_prediction, cv_confidence)
    cvaccuracy = accuracy_score(cv_label, cv_prediction)
    output = {
              'Params': params,
              'accuracy': cvaccuracy,
              'rmse': cvrmse_score,
              'Macro': {'Macro_Precision': mactest_P,
                        'Macro_Recall': mactest_R,
                        'macro_F_score': mactest_F},
              'Micro': {'Micro_Precision': mictest_P,
                        'Micro_Recall': mictest_R,
                        'micro_F_score': mictest_F},
              'Per_class': {'Pclass_Precision': test_P,
                            'Pclass_Recall': test_R,
                            'Pclass_F_score': test_F},
              'attachments': {'ID': cv_ids, 'Label': cv_label,
                              'Prediction': cv_prediction,
                              'Confidence': cv_confidence,
                              'Perfold': allfolds}
               }
    with open('output/Output_Veracity_fullPHEME.pkl', 'wb') as outfile:
        pickle.dump(output, outfile)

    return output
