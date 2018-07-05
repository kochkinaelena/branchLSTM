#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Contains the following functions: 
   
   eval_train_model - re-trains model on train+dev set and 
                      evaluates on test set
"""
import numpy
import theano
import theano.tensor as T
import lasagne
import os
import pickle
from hyperopt import STATUS_OK
from training import build_nn,iterate_minibatches
#theano.config.floatX = 'float32'
#theano.config.warn_float64 = 'raise'
#%%


def eval_train_model(params):
    print ("Retrain model on train+dev set and evaluate on testing set")
    # Initialise parameters 
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
    # Load data
    path = 'saved_data'
    brancharray = numpy.load(os.path.join(path, 'train/branch_arrays.npy'))
    num_features = numpy.shape(brancharray)[-1]
    train_mask = numpy.load(os.path.join(path,
                                         'train/mask.npy')).astype(numpy.int16)
    train_label = numpy.load(os.path.join(path, 'train/padlabel.npy'))
    
    train_rmdoublemask = numpy.load(
                            os.path.join(
                                path,
                                'train/rmdoublemask.npy')).astype(numpy.int16)
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
    # This function returns the values of the parameters of all
    # layers below one or more given Layer instances,
    # including the layer(s) itself.

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss*rmdoublesmask
    loss = lasagne.objectives.aggregate(loss, mask.flatten())
    # regularisation
    l2_penalty = l2reg * lasagne.regularization.regularize_network_params(
                                network,
                                lasagne.regularization.l2)
    loss = loss + l2_penalty

    # We could add some weight decay as well here, see lasagne.regularization.
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step.
    parameters = lasagne.layers.get_all_params(network, trainable=True)
    my_updates = lasagne.updates.adam(loss, parameters,
                                      learning_rate=learn_rate)
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function(inputs=[input_var, mask,
                                       rmdoublesmask, target_var],
                               outputs=loss,
                               updates=my_updates,
                               on_unused_input='warn')
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, mask], test_prediction,
                             on_unused_input='warn')
#%%
    # READ THE DATA
    dev_brancharray = numpy.load(os.path.join(path, 'dev/branch_arrays.npy'))
    dev_mask = numpy.load(
               os.path.join(
                   path,
                   'dev/mask.npy')).astype(numpy.int16)
    dev_label = numpy.load(os.path.join(path, 'dev/padlabel.npy'))

    dev_rmdoublemask = numpy.load(
                       os.path.join(
                        path,
                        'dev/rmdoublemask.npy')).astype(numpy.int16).flatten()

    with open(os.path.join(path,'dev/ids.pkl'), 'rb') as handle:
        dev_ids_padarray = pickle.load(handle)
    
    test_brancharray = numpy.load(os.path.join(path, 'test/branch_arrays.npy'))
    test_mask = numpy.load(
                os.path.join(
                    path,
                    'test/mask.npy')).astype(numpy.int16)

    test_rmdoublemask = numpy.load(
                os.path.join(path,
                             'test/rmdoublemask.npy')).astype(
                                                       numpy.int16).flatten()
                
    with open(os.path.join(path,'test/ids.pkl'), 'rb') as handle:
        test_ids_padarray = pickle.load(handle)

#%%
    #start training loop
    # We iterate over epochs:
    for epoch in range(num_epochs):
        #print("Epoch {} ".format(epoch))
        train_err = 0
        # In each epoch, we do a full pass over the training data:
        for batch in iterate_minibatches(brancharray, train_mask,
                                         train_rmdoublemask,
                                         train_label, mb_size,
                                         max_seq_len=25, shuffle=False):
                inputs, mask, rmdmask, targets = batch
                train_err += train_fn(inputs, mask,
                                      rmdmask, targets)
        for batch in iterate_minibatches(dev_brancharray, dev_mask,
                                         dev_rmdoublemask,
                                         dev_label, mb_size,
                                         max_seq_len=20, shuffle=False):
                inputs, mask, rmdmask, targets = batch
                train_err += train_fn(inputs, mask,
                                      rmdmask, targets)
    # And a full pass over the test data:
    test_ypred = val_fn(test_brancharray, test_mask)
    # get class label instead of probabilities
    new_test_ypred = numpy.argmax(test_ypred, axis=1).astype(numpy.int32)

    #Take mask into account
    acv_prediction = numpy.asarray(new_test_ypred)
    acv_mask = test_mask.flatten()
    clip_dev_ids = [o for o, m in zip(test_ids_padarray, acv_mask) if m == 1]
    clip_dev_prediction = [o for o, m in zip(acv_prediction, acv_mask)
                           if m == 1]
    # remove repeating instances
    uniqtwid, uindices2 = numpy.unique(clip_dev_ids, return_index=True)
    uniq_dev_prediction = [clip_dev_prediction[i] for i in uindices2]
    uniq_dev_id = [clip_dev_ids[i] for i in uindices2]
    output = {
              'status': STATUS_OK,
              'Params': params,
              'attachments': {'Predictions': pickle.dumps(uniq_dev_prediction),
                              'ID': pickle.dumps(uniq_dev_id)}
              }

    return output
