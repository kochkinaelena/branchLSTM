import os 
import json
from copy import deepcopy
from sklearn.metrics import  precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix
import pickle
from outer import convertlabeltostr
#import numpy
#%%
def tree2branches(root):
    node = root
    parent_tracker = []
    parent_tracker.append(root)
    branch = []
    branches = []
    i=0
    while True:
        node_name = node.keys()[i]
        #print node_name
        branch.append(node_name)
        # get children of the node
        first_child = node.values()[i] # actually all chldren, all tree left under this node
        if first_child != []: # if node has children
            node = first_child      # walk down
            parent_tracker.append(node)
            siblings = first_child.keys()
            i=0 # index of a current node
        else:
            branches.append(deepcopy(branch))
            i=siblings.index(node_name) # index of a current node
            while i+1>=len(siblings): # if the node does not have next siblings
                if node is parent_tracker[0]: # if it is a root node
                    return branches
                del parent_tracker[-1]
                del branch[-1]
                node = parent_tracker[-1]      # walk up ... one step
                node_name = branch[-1]
                siblings = node.keys()
                i=siblings.index(node_name)
            i=i+1    # ... walk right
            del branch[-1]

#%%
def listdir_nohidden(path):
    folds = os.listdir(path)
    newfolds = [i for i in folds if i[0] != '.']
    return newfolds


if __name__ == "__main__":

    # load labels and split for task A
    path_to_split = os.path.join('downloaded_data', 'semeval2017-task8-dataset', 'traindev')
    devfile = 'rumoureval-subtaskA-dev.json'
    with open(os.path.join(path_to_split,devfile)) as f:
        for line in f:
            dev = json.loads(line)
    trainfile = 'rumoureval-subtaskA-train.json'
    with open(os.path.join(path_to_split,trainfile)) as f:
        for line in f:
            train = json.loads(line)
    dev_tweets = dev.keys()
    train_tweets = train.keys()
    #%%
    path_to_folds = os.path.join('downloaded_data', 'semeval2017-task8-dataset', 'rumoureval-data')
    folds=listdir_nohidden(path_to_folds)
    cvfolds = {}
    allconv = []
    weird_conv = []
    weird_struct = []
    train_dev_split = {}
    train_dev_split['dev'] = []
    train_dev_split['train'] = []
    train_dev_split['test'] = []
    for nfold,fold in enumerate(folds):
        path_to_tweets = os.path.join(path_to_folds, fold)
        tweet_data =  listdir_nohidden(path_to_tweets)
        conversation = {}
        for foldr in tweet_data:
            flag = 0
            conversation['id']=foldr

            path_src = path_to_tweets+'/'+foldr+'/source-tweet'
            files_t=listdir_nohidden(path_src)
            with open(os.path.join(path_src,files_t[0])) as f:
                    for line in f:
                        src = json.loads(line)
                        src['used']=0
                        scrcid = src['id_str']
                        # add set and label to tweet info
                        # first find the tweet in one of the sets
                        # foldr - src tweet id
                        if scrcid in dev_tweets:
                            src['set'] = 'dev'
                            src['label'] = dev[scrcid]
                            flag = 'dev'
                        elif scrcid in train_tweets:
                            src['set'] = 'train'
                            src['label'] = train[scrcid]
                            flag = 'train'
                        else:
                            print ("Tweet was not found in any of the sets! ID: ",
                                   foldr)
            conversation ['source'] = src
            if src['text'] is None:
                print ("Tweet has no text", src['id'])
            tweets = []
            path_repl = path_to_tweets+'/'+foldr+'/replies'
            files_t=listdir_nohidden(path_repl)
            for repl_file in files_t:
                with open(os.path.join(path_repl, repl_file)) as f:
                    for line in f:
                        tw = json.loads(line)
                        tw['used']=0
                        replyid = tw['id_str']
                        if replyid in dev_tweets:
                            tw['set'] = 'dev'
                            tw['label'] = dev[replyid]
    #                        train_dev_tweets['dev'].append(tw)
                            if flag=='train':
                                print ("On no! The tree is split between sets",
                                       foldr)
                        elif replyid in train_tweets:
                            tw['set'] = 'train'
                            tw['label'] = train[replyid]
    #                        train_dev_tweets['train'].append(tw)
                            if flag=='dev':
                                print ("On no! The tree is split between sets",
                                       foldr)
                        else:
                            print ("Tweet was not found in any of the sets! ID: ",
                                   foldr)
                        tweets.append(tw)
                        if tw['text'] is None:
                            print ("Tweet has no text", tw['id'])
            conversation['replies'] = tweets
            path_struct = path_to_tweets+'/'+foldr+'/structure.json'
            with open(path_struct) as f:
                    for line in f:
                        struct = json.loads(line)

            if len(struct)>1:
    #            print "Structure has more than one root",
                new_struct = {}
                new_struct[foldr] = struct[foldr]
                struct = new_struct
                weird_conv.append(conversation.copy())
                weird_struct.append(struct)
                # Take the item from the strucutre that's key is same as source tweet id

            conversation['structure'] = struct
            branches = tree2branches(conversation['structure'])
            conversation['branches'] = branches
            train_dev_split[flag].append(conversation.copy())
            allconv.append(conversation.copy())
        cvfolds[fold] = allconv
        allconv = []
    #%%
    # read testing data
    path_to_test = os.path.join('downloaded_data', 'semeval2017-task8-test-data')
    test_folders = listdir_nohidden(path_to_test)
    conversation = {}
    for tfldr in test_folders:
        conversation['id']=tfldr
        path_src = path_to_test+'/'+tfldr+'/source-tweet'
        files_t=listdir_nohidden(path_src)
        with open(os.path.join(path_src,files_t[0])) as f:
            for line in f:
                src = json.loads(line)
                src['used']=0
        conversation ['source'] = src
        tweets = []
        path_repl = path_to_test+'/'+tfldr+'/replies'
        files_t=listdir_nohidden(path_repl)
        for repl_file in files_t:
            with open(os.path.join(path_repl, repl_file)) as f:
                for line in f:
                    tw = json.loads(line)
                    tw['used']=0
            tweets.append(tw)
        conversation['replies'] = tweets
        path_struct = path_to_test+'/'+tfldr+'/structure.json'
        with open(path_struct) as f:
            for line in f:
                struct = json.loads(line)
        conversation['structure'] = struct
        branches = tree2branches(conversation['structure'])
        conversation['branches'] = branches
        train_dev_split['test'].append(conversation.copy())

    #%%
    path_to_split = os.path.join('downloaded_data', 'semeval2017-task8-dataset', 'traindev')
    devfile = 'rumoureval-subtaskA-dev.json'
    with open(os.path.join(path_to_split,devfile)) as f:
        for line in f:
            dev = json.loads(line)
    trainfile = 'rumoureval-subtaskA-train.json'
    with open(os.path.join(path_to_split,trainfile)) as f:
        for line in f:
            train = json.loads(line)
    dev_tweets = dev.keys()
    train_tweets = train.keys()

    testfile = 'subtaska.json'
    with open(testfile) as f:
        for line in f:
            test_truevals = json.loads(line)
    test_tweets = test_truevals.keys()

    # Read prediction json dict
    submission_file = os.path.join("output", "predictions.txt")
    submission = json.load(open(submission_file, 'r'))
    testfile = 'subtaska.json'
    with open(testfile) as f:
        for line in f:
            test_truevals = json.loads(line)
    #%%
    # use train_dev_split from preprocessing

    alltestinfo = train_dev_split['test']

    alltestbranches = []
    # get all test branches out of it
    for indx, element in enumerate(alltestinfo):
        alltestbranches.extend(element['branches'])
    # loop over each tweet in testing set and find its depth to create id: depth dictionary
    depthinfo = {}
    for tweetid in submission.keys():
        for branch in alltestbranches:
            if tweetid in branch:
                depthinfo[tweetid] = branch.index(tweetid)


    # group true labels and predictions according to their depth
    #%%
    # depthinfo id: depth

    # submission id: prediction

    # test_truva; id: label

    depth_groups = {}

    depth_groups['0'] = []
    depth_groups['1'] = []
    depth_groups['2'] = []
    depth_groups['3'] = []
    depth_groups['4'] = []
    depth_groups['5'] = []
    depth_groups['6+'] = []


    # find all keys in that depth group

    for tweetid, tweetdepth in depthinfo.iteritems():
        if tweetdepth == 0:
            depth_groups['0'].append(tweetid)
        elif tweetdepth == 1:
            depth_groups['1'].append(tweetid)
        elif tweetdepth == 2:
            depth_groups['2'].append(tweetid)
        elif tweetdepth == 3:
            depth_groups['3'].append(tweetid)
        elif tweetdepth == 4:
            depth_groups['4'].append(tweetid)
        elif tweetdepth == 5:
            depth_groups['5'].append(tweetid)
        elif tweetdepth >5:
            depth_groups['6+'].append(tweetid)

    ## make a list

    depth_predictions = {}

    depth_predictions['0'] = []
    depth_predictions['1'] = []
    depth_predictions['2'] = []
    depth_predictions['3'] = []
    depth_predictions['4'] = []
    depth_predictions['5'] = []
    depth_predictions['6+'] = []

    depth_labels = {}

    depth_labels['0'] = []
    depth_labels['1'] = []
    depth_labels['2'] = []
    depth_labels['3'] = []
    depth_labels['4'] = []
    depth_labels['5'] = []
    depth_labels['6+'] = []

    depth_result = {}

    for depthgr in depth_groups.keys():
        depth_predictions[depthgr] = [submission[x] for x in depth_groups[depthgr]]
        depth_labels[depthgr] = [test_truevals[x] for x in depth_groups[depthgr]]

        test1 = [x for x in depth_groups[depthgr]]
        test2 = [x for x in depth_groups[depthgr]]

        _, _, mactest_F, _ = precision_recall_fscore_support(depth_labels[depthgr],
                                                             depth_predictions[depthgr],
                                                             average='macro')
        _, _, mictest_F, _ = precision_recall_fscore_support(depth_labels[depthgr],
                                                             depth_predictions[depthgr],
                                                             average='micro')
        _, _, test_F, _ = precision_recall_fscore_support(depth_labels[depthgr],
                                                          depth_predictions[depthgr])

        depth_result[depthgr] = [mactest_F, mictest_F, test_F]
    print("Table 4: Number of tweets per depth and performance at each of the depths.")
    print(depth_result)

    #%%
    print ("Table 5: Confusion matrix")

    true = []
    pred = []

    for k in test_truevals.keys():

        true.append(test_truevals[k])
        pred.append(submission[k])

    print (confusion_matrix(true,pred))

    # Define some useful labels for table rows and columns
    labels = ("Precision", "Recall", "F-score", "Support")
    class_labels = ("Support", "Deny", "Query", "Comment")
    class_labels_gap = ("",) + class_labels

    print "\n\n--- Table 3 ---"

    print "\nPart 1: Results on testing set"

    print "\nAccuracy =", accuracy_score(true, pred)

    print "\nMacro-average:"
    macroavg_prfs = precision_recall_fscore_support(true, pred, average='macro')
    for lab, val in zip(labels, macroavg_prfs):
        if val is not None:
            print "%-12s%-12.4f" % (lab, val)
        else:
            print "%-12s%-12s" % (lab, "--")

    print "\nPer-class:"
    perclass_prfs = precision_recall_fscore_support(true, pred)
    print "%-12s%-12s%-12s%-12s%-12s" % class_labels_gap
    for lab, vals in zip(labels, perclass_prfs):
        if lab is "Support":
            print "%-12s%-12i%-12i%-12i%-12i" % (lab, vals[0], vals[1], vals[2], vals[3])
        else:
            print "%-12s%-12.4f%-12.4f%-12.4f%-12.4f" % (lab, vals[0], vals[1], vals[2], vals[3])

    print "\nPart 2: Results on development set"
    trials_file = os.path.join("output", "trials.txt")

    if os.path.exists(trials_file):
        with open(trials_file, 'rb') as f:
           trials = pickle.load(f)
        #extract results for dev data
        dev_result_id = pickle.loads(trials.attachments['ATTACH::60::ID']) # 60 is best trial number
        dev_result_labels = pickle.loads(trials.attachments['ATTACH::60::Labels'])
        dev_result_predictions = pickle.loads(trials.attachments['ATTACH::60::Predictions'])
        # change id format from float to int or str
        strpred = [convertlabeltostr(s) for s in dev_result_predictions] #extract results for test data
        # Transform to submission format
        dev_results_dict = dict(zip(dev_result_id, strpred))
        import json
        with open('result_dev.json', 'w') as outfile:
           json.dump(dev_results_dict, outfile)

        with open('result_dev_GN.json', 'r') as outfile:
            dev_predictions = json.load(outfile)

        devtrue = []
        devpred = []

        for k in dev.keys():

            devtrue.append(dev[k])
            devpred.append(dev_predictions[k])
        print ("Table 3: Results on development set")
        print ("Accuracy ", accuracy_score(devtrue, devpred))
        print ("Macroavg precision, recall, fscore, support = ", precision_recall_fscore_support(devtrue, devpred, average='macro'))
        print ("Per class precision, recall, fscore, support = ", precision_recall_fscore_support(devtrue, devpred))
    else:
        print "\nCould not find trials.txt; unable to generate results for development set in Table 3.\n"
