"""
This code is to load RumEval dataset
"""
import os
import json


def read_RumEv():
    # load labels and split for task A
    path_to_split = 'data/semeval2017-task8-dataset/traindev'
    devfileA = 'rumoureval-subtaskA-dev.json'
    with open(os.path.join(path_to_split, devfileA)) as f:
        for line in f:
            devA = json.loads(line)
    trainfileA = 'rumoureval-subtaskA-train.json'
    with open(os.path.join(path_to_split, trainfileA)) as f:
        for line in f:
            trainA = json.loads(line)
    path_to_test_lables = "data/subtaska.json"
    with open(path_to_test_lables) as f:
        for line in f:
            testA = json.loads(line)
    # for task B (the following code won't be directly suitable for B)
    path_to_split = 'data/semeval2017-task8-dataset/traindev'
    devfile = 'rumoureval-subtaskB-dev.json'
    with open(os.path.join(path_to_split, devfile)) as f:
        for line in f:
            dev = json.loads(line)
    trainfile = 'rumoureval-subtaskB-train.json'
    with open(os.path.join(path_to_split, trainfile)) as f:
        for line in f:
            train = json.loads(line)
    dev_tweets = dev.keys()
    train_tweets = train.keys()
    path_to_test_lables = "data/subtaskb.json"
    with open(path_to_test_lables) as f:
        for line in f:
            test = json.loads(line)
    # load folds and conversations
    path_to_folds = 'data/semeval2017-task8-dataset/rumoureval-data'
    folds = os.listdir(path_to_folds)
    newfolds = [i for i in folds if i[0] != '.']
    folds = newfolds
    cvfolds = {}
    allconv = []
    weird_conv = []
    weird_struct = []
    train_dev_split = {}
    train_dev_split['dev'] = []
    train_dev_split['train'] = []
    train_dev_split['test'] = []
    for nfold, fold in enumerate(folds):
        path_to_tweets = os.path.join(path_to_folds, fold)
        tweet_data = os.listdir(path_to_tweets)
        newfolds = [i for i in tweet_data if i[0] != '.']
        tweet_data = newfolds
        conversation = {}
        for foldr in tweet_data:
            # which set conversation belongs to
            conversation['id'] = foldr
            if foldr in dev_tweets:
                convset = 'dev'
                # find its label
                conversation['veracity'] = dev[foldr]
            elif foldr in train_tweets:
                convset = "train"
                conversation['veracity'] = train[foldr]
            else:
                print("Conversation  not found", foldr)
            path_src = path_to_tweets+'/'+foldr+'/source-tweet'
            files_t = os.listdir(path_src)
            newfolds = [i for i in files_t if i[0] != '.']
            files_t = newfolds
            with open(os.path.join(path_src, files_t[0])) as f:
                    for line in f:
                        src = json.loads(line)
                        src['used'] = 0
                        scrcid = src['id_str']
                        src['set'] = convset
                        src['label'] = conversation['veracity']
                        src['conv_id'] = foldr
    #                    convset  - which set it belongs to
    #                    scrcid - string id of tweet can be key in setA
                        if convset == 'dev':
                            stance = devA[scrcid]
                        elif convset == 'train':
                            stance = trainA[scrcid]
                        else:
                            print ("Tweet is not found in dev or train")
                        src['stance'] = stance
            conversation['source'] = src
            if src['text'] is None:
                print( "Tweet has no text", src['id'])
            tweets = []
            path_repl = path_to_tweets+'/'+foldr+'/replies'
            files_t = os.listdir(path_repl)
            if len(files_t) < 1:
                print ("No replies", foldr)
            newfolds = [i for i in files_t if i[0] != '.']
            files_t = newfolds
            for repl_file in files_t:
                with open(os.path.join(path_repl, repl_file)) as f:
                    for line in f:
                        tw = json.loads(line)
                        tw['used'] = 0
                        replyid = tw['id_str']
                        tw['set'] = convset
                        tw['label'] = conversation['veracity']
                        tw['conv_id'] = foldr
                        if convset == 'dev':
                            stance = devA[replyid]
                        elif convset == 'train':
                            stance = trainA[replyid]
                        else:
                            print ("Tweet is not found in dev or train")
                        tw['stance'] = stance
                        tweets.append(tw)
                        if tw['text'] is None:
                            print ("Tweet has no text", tw['id'])
            conversation['replies'] = tweets
            path_struct = path_to_tweets+'/'+foldr+'/structure.json'
            with open(path_struct) as f:
                    for line in f:
                        struct = json.loads(line)
            if len(struct) > 1:
                print ("Structure has more than one root",conversation['id'])
                new_struct = {}
                new_struct[foldr] = struct[foldr]
                struct = new_struct
                weird_conv.append(conversation.copy())
                weird_struct.append(struct)
            conversation['structure'] = struct
            train_dev_split[convset].append(conversation.copy())
            allconv.append(conversation.copy())
        cvfolds[fold] = allconv
        allconv = []
    # read testing data
    path_to_test = 'data/semeval2017-task8-test-data'
    test_folders = os.listdir(path_to_test)
    newfolds = [i for i in test_folders if i[0] != '.']
    test_folders = newfolds
    conversation = {}
    for tfldr in test_folders:
        if tfldr in test.keys():
            conversation['id'] = tfldr
            conversation['veracity'] = test[tfldr]
            path_src = path_to_test+'/'+tfldr+'/source-tweet'
            files_t = os.listdir(path_src)
            with open(os.path.join(path_src, files_t[0])) as f:
                for line in f:
                    src = json.loads(line)
                    src['used'] = 0
                    stance = testA[src['id_str']]
                    src['stance'] = stance

            conversation['source'] = src
            tweets = []
            path_repl = path_to_test+'/'+tfldr+'/replies'
            files_t = os.listdir(path_repl)
            newfolds = [i for i in files_t if i[0] != '.']
            files_t = newfolds
            for repl_file in files_t:
                with open(os.path.join(path_repl, repl_file)) as f:
                    for line in f:
                        tw = json.loads(line)
                        tw['used'] = 0
                        if tw['id_str'] in testA.keys():
                            stance = testA[tw['id_str']]
                            tw['stance'] = stance
                        else:
                            print (tw['id_str'])
                tweets.append(tw)
            conversation['replies'] = tweets
            path_struct = path_to_test+'/'+tfldr+'/structure.json'
            with open(path_struct) as f:
                for line in f:
                    struct = json.loads(line)
            conversation['structure'] = struct
            train_dev_split['test'].append(conversation.copy())
    return train_dev_split
