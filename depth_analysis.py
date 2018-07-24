import os
import json
from copy import deepcopy
from sklearn.metrics import  precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix
import pickle
from outer import convertlabeltostr
import numpy

from preprocessing import tree2branches

import matplotlib
if "Darwin" in os.uname():
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def listdir_nohidden(path):
    folds = os.listdir(path)
    newfolds = [i for i in folds if i[0] != '.']
    return newfolds


def load_data():

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

    # Read the predictions of the model
    submission_file = os.path.join("output", "predictions.txt")
    submission = json.load(open(submission_file, 'r'))

    # And then the corresponding test data
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

    return depthinfo, submission, test_truevals, dev


def load_trials_data():

    trials_file = os.path.join("output", "trials.txt")

    if os.path.exists(trials_file):
        # Load the trials data
        with open(trials_file, 'rb') as f:
            trials = pickle.load(f)

        # We need to examine the best trial
        best_trial_id = trials.best_trial["tid"]
        best_trial_loss = trials.best_trial["result"]["loss"]
        dev_result_id = pickle.loads(trials.attachments["ATTACH::%d::ID" % best_trial_id])
        dev_result_predictions = pickle.loads(trials.attachments["ATTACH::%d::Predictions" % best_trial_id])

        # Change ID format from ints to strings
        strpred = [convertlabeltostr(s) for s in dev_result_predictions]

        # Transform to submission format and save
        dev_results_dict = dict(zip(dev_result_id, strpred))
        with open("result_dev.json", "w") as outfile:
            json.dump(dev_results_dict, outfile)
    else:
        best_trial_id = None
        best_trial_loss = None
        dev_results_dict = None
        print "trials.txt is not available"

    return best_trial_id, best_trial_loss, dev_results_dict


def get_true_and_predicted_classes(true_tweet_classes, predicted_tweet_classes):

    # Sometimes this function may be called with empty predicted classes (e.g. if trials is not available).
    # In that case, return None for the true and predicted lists.
    if predicted_tweet_classes is None:
        true = None
        pred = None
        return true, pred

    true = []
    pred = []

    # Generate lists of true and predicted classes for all tweets in this set
    for k in true_tweet_classes.keys():
        true.append(true_tweet_classes[k])
        pred.append(predicted_tweet_classes[k])

    return true, pred


def calculate_results_at_each_depth(depthinfo, submission, test_truevals):

    # Group true labels and predictions according to their depth
    # depthinfo id: depth
    # submission id: prediction
    # test_truvals id: label

    depth_groups = {}
    depth_groups['0'] = []
    depth_groups['1'] = []
    depth_groups['2'] = []
    depth_groups['3'] = []
    depth_groups['4'] = []
    depth_groups['5'] = []
    depth_groups['6+'] = []


    # Find all keys in that depth group
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

    # make a list

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

        _, _, mactest_F, _ = precision_recall_fscore_support(depth_labels[depthgr],
                                                             depth_predictions[depthgr],
                                                             average='macro')
        _, _, mictest_F, _ = precision_recall_fscore_support(depth_labels[depthgr],
                                                             depth_predictions[depthgr],
                                                             average='micro')
        _, _, test_F, _ = precision_recall_fscore_support(depth_labels[depthgr],
                                                          depth_predictions[depthgr])

        depth_result[depthgr] = [mactest_F, mictest_F, test_F]

    return depth_labels, depth_result


def print_table_three(true, pred, devtrue, devpred, best_trial_id, best_trial_loss):

    print "\n\n--- Table 3 ---"

    print "\nPart 1: Results on testing set"

    print "\nAccuracy =", accuracy_score(true, pred)

    print "\nMacro-average:"
    macroavg_prfs = precision_recall_fscore_support(true, pred, average='macro')
    for lab, val in zip(results_headers, macroavg_prfs):
        if val is not None:
            print "%-12s%-12.4f" % (lab, val)
        else:
            print "%-12s%-12s" % (lab, "--")

    print "\nPer-class:"
    perclass_prfs = precision_recall_fscore_support(true, pred)
    print "%-12s%-12s%-12s%-12s%-12s" % class_labels_gap
    for lab, vals in zip(results_headers, perclass_prfs):
        if lab is "Support":
            print "%-12s%-12i%-12i%-12i%-12i" % (lab, vals[0], vals[1], vals[2], vals[3])
        else:
            print "%-12s%-12.4f%-12.4f%-12.4f%-12.4f" % (lab, vals[0], vals[1], vals[2], vals[3])

    print "\nPart 2: Results on development set"

    if best_trial_id is not None:

        print "\nBest trial =", best_trial_id, "with loss", best_trial_loss

        #  Output is in the same format as the earlier part of Table 3
        print "Accuracy =", accuracy_score(devtrue, devpred)

        print "\nMacro-average:"
        dev_macroavg_prfs = precision_recall_fscore_support(devtrue, devpred, average='macro')
        for lab, val in zip(results_headers, dev_macroavg_prfs):
            if val is not None:
                print "%-12s%-12.4f" % (lab, val)
            else:
                print "%-12s%-12s" % (lab, "--")

        print "\nPer-class:"
        dev_perclass_prfs = precision_recall_fscore_support(devtrue, devpred)
        print "%-12s%-12s%-12s%-12s%-12s" % class_labels_gap
        for lab, vals in zip(results_headers, dev_perclass_prfs):
            if lab is "Support":
                print "%-12s%-12i%-12i%-12i%-12i" % (lab, vals[0], vals[1], vals[2], vals[3])
            else:
                print "%-12s%-12.4f%-12.4f%-12.4f%-12.4f" % (lab, vals[0], vals[1], vals[2], vals[3])

    # trials.txt is generated by the parameter search, and so won't be generated by outer.py unless requested
    # (substantial compute resources needed). So we may not be able to generate this part of the table.
    else:
        print "\nCould not find trials.txt; unable to generate results for development set in Table 3.\n"


def print_extra_details(best_trial_id):

    trials = pickle.load(open(os.path.join("output", "trials.txt"), "rb"))

    # Print out the best combination of hyperparameters
    print "\n--- New Table ---\n"
    print "The best combination of hyperparameters, found in trial " + str(best_trial_id) + ", was:"
    for param, param_value in trials.best_trial["result"]["Params"].iteritems():
        print "\t", param, "=", param_value

    # Let's examine the loss function at each iteration of the hyperparameter tuning process
    print "\n--- New Figure ---"

    # Extract the loss values from the full list of results, and calculate the running minimum value
    loss = numpy.asarray([r["loss"] for r in trials.results])
    running_min_loss = numpy.minimum.accumulate(loss)
    lowest_loss = loss[best_trial_id]
    all_best_ids = numpy.where(loss == lowest_loss)[0]

    # Plot the loss and running loss values against the iteration number, and save to the output folder
    plt.plot(range(0, len(loss)), loss, label="loss")
    plt.plot(range(0, len(running_min_loss)), running_min_loss, label="running min(loss)")
    plt.plot(best_trial_id, lowest_loss, "ro", label="min(loss)")
    if len(all_best_ids) > 1:
        plt.plot(all_best_ids, lowest_loss*numpy.ones(all_best_ids.shape), "rx", label="repeated min(loss)")
    plt.legend()
    plt.title("Hyperparameter optimisation")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(os.path.join("output", "hyperparameter_loss_values.pdf"))

    # Give details of other hyperparameter combinations that also achieved this loss
    if len(all_best_ids) > 1:
        print "\nWARNING: multiple hyperparameter combinations achieved the same lowest loss value as trial", best_trial_id
        print "ID               ",
        for id in all_best_ids:
            print "%-17d" % id,
        print ""
        for param in trials.results[all_best_ids[0]]["Params"]:
            print "%-17s" % param,
            for id in all_best_ids:
                print "%-17.4f" % trials.results[id]["Params"][param],
            print ""

    print "\nFigure showing hyperparameter optimisation progress can be found in the output folder.\n"


def print_table_four(depth_labels, depth_result):

    print "\n\n--- Table 4 ---"
    print "\nNumber of tweets per depth and performance at each of the depths\n"

    # Print the column headers
    table_four_headers = ("Depth", "# tweets", "# Support", "# Deny", "# Query", "# Comment", "Accuracy", "MacroF") + class_labels
    for col in table_four_headers:
        print "%-11s" % col,
    print ""

    #  Print results in depth level order
    for depth in sorted(depth_result):

        # Work out which class the accuracy values refer to (precision_recall_fscore_support() outputs values in the
        # sorted order of the unique classes of tweets at that depth)
        depth_class_accuracy = depth_result[depth][2]
        depth_class_labels = sorted(set(depth_labels[depth]))

        # Print the depth and classes of tweets at that depth
        print "%-12s%-11i" % (depth, len(depth_labels[depth])),
        for lab in class_labels:
            print "%-11i" % depth_labels[depth].count(lab.lower()),

        # Print the accuracy, macro-F and class-specific performance at each depth
        print "%-12.4f%-11.4f" % \
              (depth_result[depth][1], depth_result[depth][0]),
        for lab in class_labels:
            if lab.lower() in depth_class_labels:
                class_ind = depth_class_labels.index(lab.lower())
                print "%-11.4f" % depth_class_accuracy[class_ind],
            else:
                print "%-11.4f" % 0.0,
        print ""


def print_table_five(true, pred):

    print "\n\n--- Table 5 ---"
    print "\nConfusion matrix\n"

    # Generate the confusion matrix and the list of labels (as above, in sorted class order as long as each class
    # appears once, which they all do).
    conf_mat = confusion_matrix(true, pred)
    class_labels_mat = ("Lab \\ Pred",) + tuple(sorted(class_labels))

    # Print the header and then the confusion matrix
    print "%-12s%-12s%-12s%-12s%-12s" % class_labels_mat
    for lab, conf_row in zip(sorted(class_labels), conf_mat):
        row = (lab,) + tuple(conf_row)
        print "%-12s%-12i%-12i%-12i%-12i" % row


if __name__ == "__main__":

    # From the supplied files, load the tweet depth, predicted (test) labels, test labels and development labels.
    tweet_depth, test_predicted_labels, test_labels, dev_labels = load_data()

    # If it is present, load data from trials file and format in the same way as the submitted files
    # (return None if the trials file is not available)
    best_trial, best_loss, dev_predicted_labels = load_trials_data()

    # Analyse the results separately at each depth
    level_for_each_depth, results_for_each_depth = \
        calculate_results_at_each_depth(tweet_depth, test_predicted_labels, test_labels)

    # Get lists of the true and predicted classes for the test and, if possible, development sets
    true_labels_test, predicted_labels_test = get_true_and_predicted_classes(test_labels, test_predicted_labels)
    true_labels_dev, predicted_labels_dev = get_true_and_predicted_classes(dev_labels, dev_predicted_labels)

    # Define some useful labels for table rows and columns
    results_headers = ("Precision", "Recall", "F-score", "Support")
    class_labels = ("Support", "Deny", "Query", "Comment")
    class_labels_gap = ("",) + class_labels

    # Print the tables
    print_table_four(level_for_each_depth, results_for_each_depth)
    print_table_five(true_labels_test, predicted_labels_test)
    print_table_three(true_labels_test, predicted_labels_test,
                      true_labels_dev, predicted_labels_dev, best_trial, best_loss)

    # If the trials file is available, output more details of the best hyperparameter combinations and prepare a figure
    # showing the loss during the hyperparameter choice process
    if best_trial is not None:
        print_extra_details(best_trial)
