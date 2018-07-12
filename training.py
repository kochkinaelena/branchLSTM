#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Contains the following functions: 
   
build_nn - defines architecture of Neural Network
iterate_minibatches - splits training data into mini-batches and 
                     returns iterator
objective_train_model - trains model on the training set, 
                       evaluates on development set and returns output 
                       in the format suitable for use with hyperopt package 
"""
import numpy
import theano
import theano.tensor as T
import lasagne
from sklearn.metrics import precision_recall_fscore_support
import os
import timeit
import pickle
from hyperopt import STATUS_OK
from sklearn.metrics import accuracy_score
#theano.config.floatX = 'float32'
#theano.config.warn_float64 = 'raise'

#%%


def build_nn(input_var, mask, num_features, num_lstm_layers, num_lstm_units,
             num_dense_layers, num_dense_units):

    l_input = lasagne.layers.InputLayer(shape=(None, None, num_features),
                                        input_var=input_var)

    l_mask = lasagne.layers.InputLayer(shape=(None, None), input_var=mask)

    l_lstm1 = lasagne.layers.LSTMLayer(
                    l_input, 
                    num_units=num_lstm_units,
                    mask_input=l_mask,
                    peepholes=False,
                    forgetgate=lasagne.layers.Gate(
                        W_in=lasagne.init.Normal(0.1),
                        W_hid=lasagne.init.Normal(0.1),
                        W_cell=lasagne.init.Normal(0.1),
                        b=lasagne.init. Constant(2.),
                        nonlinearity=lasagne.nonlinearities.sigmoid))
    for nlstm in range(num_lstm_layers-1):
        l_lstm2 = lasagne.layers.LSTMLayer(
                  l_lstm1,
                  num_units=num_lstm_units,
                  mask_input=l_mask, peepholes=False,
                  forgetgate=lasagne.layers.Gate(
                        W_in=lasagne.init.Normal(0.1),
                        W_hid=lasagne.init.Normal(0.1),
                        W_cell=lasagne.init.Normal(0.1),
                        b=lasagne.init. Constant(2.),
                        nonlinearity=lasagne.nonlinearities.sigmoid))
        l_lstm1 = l_lstm2

    l_shp = lasagne.layers.ReshapeLayer(l_lstm1, (-1, num_lstm_units))

    l_in_drop = lasagne.layers.DropoutLayer(l_shp, p=0.2)

    l_hidden1 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
                                  l_in_drop,
                                  num_units=num_dense_units,
                                  W=lasagne.init.Normal(0.1),
                                  b=lasagne.init.Normal(0.1),
                                  nonlinearity=lasagne.nonlinearities.rectify))

    for nl in range(num_dense_layers-1):
        l_hidden2 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
                                  l_hidden1,
                                  num_units=num_dense_units,
                                  W=lasagne.init.Normal(0.1),
                                  b=lasagne.init.Normal(0.1),
                                  nonlinearity=lasagne.nonlinearities.rectify))
        l_hidden1 = l_hidden2

    l_out_drop = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)

    l_softmax = lasagne.layers.DenseLayer(
                                   l_out_drop,
                                   num_units=4,
                                   W=lasagne.init.Normal(0.1),
                                   b=lasagne.init.Normal(0.1),
                                   nonlinearity=lasagne.nonlinearities.softmax)

    return (l_softmax)
#%%


def iterate_minibatches(inputs, mask, rmdoublemask, targets,
                        batchsize, max_seq_len=25, shuffle=False):

    targets = numpy.reshape(targets, (-1, max_seq_len))
    rmdoublemask = numpy.reshape(rmdoublemask, (-1, max_seq_len))
    assert len(inputs) == len(targets)
    if shuffle:
        indices = numpy.arange(len(inputs))
        numpy.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        newtargets = targets[excerpt].flatten()
        newrmdoublemask = rmdoublemask[excerpt].flatten()
        yield inputs[excerpt], mask[excerpt], newrmdoublemask, newtargets
#%%


def objective_train_model(params):
    # Initialise parameters 
    start = timeit.default_timer()
    print(params)
    num_lstm_units = int(params['num_lstm_units'])
    num_lstm_layers = int(params['num_lstm_layers'])
    num_dense_layers = int(params['num_dense_layers'])
    num_dense_units = int(params['num_dense_units'])
    num_epochs = params['num_epochs']
    learn_rate = params['learn_rate']
    mb_size = params['mb_size']
    l2reg = params['l2reg']
    rng_seed = params['rng_seed']
#%%
    # Load training data
    path = 'saved_data'
    brancharray = numpy.load(os.path.join(path, 'train/branch_arrays.npy'))
    num_features = numpy.shape(brancharray)[-1]
    train_mask = numpy.load(
            os.path.join(path, 'train/mask.npy')).astype(numpy.int16)
    train_label = numpy.load(
            os.path.join(path, 'train/padlabel.npy'))
    train_rmdoublemask = numpy.load(
            os.path.join(path, 'train/rmdoublemask.npy')).astype(numpy.int16)
    train_rmdoublemask = train_rmdoublemask.flatten()
#%%
    numpy.random.seed(rng_seed)
    rng_inst = numpy.random.RandomState(rng_seed)
    lasagne.random.set_rng(rng_inst)
    input_var = T.ftensor3('inputs')
    mask = T.wmatrix('mask')
    target_var = T.ivector('targets')
    rmdoublesmask = T.wvector('rmdoublemask')
    # Build network
    network = build_nn(input_var, mask, num_features,
                       num_lstm_layers=num_lstm_layers,
                       num_lstm_units=num_lstm_units,
                       num_dense_layers=num_dense_layers,
                       num_dense_units=num_dense_units)
    # This function returns the values of the parameters
    # of all layers below one or more given Layer instances,
    # including the layer(s) itself.

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):

    prediction = lasagne.layers.get_output(network)

    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss*rmdoublesmask
    loss = lasagne.objectives.aggregate(loss, mask.flatten())
    # regularisation

    l2_penalty = l2reg * lasagne.regularization.regularize_network_params(
                                            network, lasagne.regularization.l2)
    loss = loss + l2_penalty

    # We could add some weight decay as well here, see lasagne.regularization.
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Adadelta
    parameters = lasagne.layers.get_all_params(network, trainable=True)
    my_updates = lasagne.updates.adam(loss, parameters,
                                      learning_rate=learn_rate)
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    test_loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                            target_var)
    test_loss = test_loss*rmdoublesmask
    test_loss = lasagne.objectives.aggregate(test_loss, mask.flatten())

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function(inputs=[input_var, mask,
                                       rmdoublesmask, target_var],
                               outputs=loss,
                               updates=my_updates, on_unused_input='warn')

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, mask, rmdoublesmask,
                             target_var], [test_loss, test_prediction],
                             on_unused_input='warn')
#%%
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # print("Epoch {} ".format(epoch))
        train_err = 0
        # In each epoch, we do a full pass over the training data:
        for batch in iterate_minibatches(brancharray, train_mask,
                                         train_rmdoublemask,
                                         train_label, mb_size,
                                         shuffle=False):
                inputs, mask, rmdmask, targets = batch
                train_err += train_fn(inputs, mask,
                                      rmdmask, targets)
        

#%%
    # Load development data
    dev_brancharray = numpy.load(os.path.join(path, 'dev/branch_arrays.npy'))
    dev_mask = numpy.load(
               os.path.join(path, 'dev/mask.npy')).astype(numpy.int16)
    dev_label = numpy.load(os.path.join(path, 'dev/padlabel.npy'))
    
    dev_rmdoublemask = numpy.load(os.path.join(
                   path, 'dev/rmdoublemask.npy')
                   ).astype(numpy.int16).flatten()

    
    with open(os.path.join(path, 'dev/ids.pkl'), 'rb') as handle:
        dev_ids_padarray = pickle.load(handle)

#%%
    # get predictions for development set
    err, val_ypred = val_fn(dev_brancharray, dev_mask,
                            dev_rmdoublemask, dev_label.flatten())
    val_ypred = numpy.argmax(val_ypred, axis=1).astype(numpy.int32)

    acv_label = dev_label.flatten()
    acv_prediction = numpy.asarray(val_ypred)
    acv_mask = dev_mask.flatten()
    clip_dev_label = [o for o, m in zip(acv_label, acv_mask) if m == 1]
    clip_dev_ids = [o for o, m in zip(dev_ids_padarray, acv_mask) if m == 1]
    clip_dev_prediction = [o for o, m in zip(acv_prediction, acv_mask)
                           if m == 1]
    # remove repeating instances
    uniqtwid, uindices2 = numpy.unique(clip_dev_ids, return_index=True)
    uniq_dev_label = [clip_dev_label[i] for i in uindices2]
    uniq_dev_prediction = [clip_dev_prediction[i] for i in uindices2]
    uniq_dev_id = [clip_dev_ids[i] for i in uindices2]
    dev_accuracy = accuracy_score(uniq_dev_label, uniq_dev_prediction)
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
    # to change scoring objective you need to change 'loss'
    output = {'loss': 1-dev_accuracy,
              'status': STATUS_OK,
              'Params': params,
              'Macro': {'Macro_Precision': mactest_P,
                        'Macro_Recall': mactest_R,
                        'macro_F_score': mactest_F},
              'Micro': {'Micro_Precision': mictest_P,
                        'Micro_Recall': mictest_R,
                        'micro_F_score': mictest_F},
              'Support': {'Support_Precision': test_P[0],
                          'Support_Recall': test_R[0],
                          'Support_F_score': test_F[0]},
              'Comment': {'Comment_Precision': test_P[1],
                          'Comment_Recall': test_R[1],
                          'Comment_F_score': test_F[1]},
              'Deny': {'Deny_Precision': test_P[2],
                       'Deny_Recall': test_R[2],
                       'Deny_F_score': test_F[2]},
              'Appeal': {'Appeal_Precision': test_P[3],
                         'Appeal_Recall': test_R[3],
                         'Appeal_F_score': test_F[3]},
              'attachments': {'Labels': pickle.dumps(uniq_dev_label),
                              'Predictions': pickle.dumps(uniq_dev_prediction),
                              'ID': pickle.dumps(uniq_dev_id)}
              }

    print("1-accuracy loss = ", output['loss'])

    stop = timeit.default_timer()
    print("Time: ", stop - start)
    return output
