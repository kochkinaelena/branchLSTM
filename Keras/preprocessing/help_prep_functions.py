"""
This file contains utility functions for preprocessing
"""
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from copy import deepcopy
import gensim
# CONVERT ANNOTATIONS ACCORDING TO SCHEME:
# - misinfo 0, true 1 -> true
# - misinfo 1, true 0 -> false
# - misinfo 0, true 0 -> unverified
# - misinfo only means true=0
# They can never be both 1.

def convert_annotations(annotation, string=True):
    if 'misinformation' in annotation.keys() and 'true'in annotation.keys():
        if (int(annotation['misinformation']) == 0) and (
                            int(annotation['true']) == 0):
            if string:
                label = "unverified"
            else:
                label = 2
        elif (int(annotation['misinformation']) == 0) and (
                              int(annotation['true']) == 1):
            if string:
                label = "true"
            else:
                label = 1
        elif (int(annotation['misinformation']) == 1) and (
                              int(annotation['true']) == 0):
            if string:
                label = "false"
            else:
                label = 0
        elif (int(annotation['misinformation']) == 1) and (
                              int(annotation['true']) == 1):
            print ("OMG! They both are 1!")
            print(annotation['misinformation'])
            print(annotation['true'])
            label = None
    elif ('misinformation' in annotation.keys()) and (
                      'true' not in annotation.keys()):
        if int(annotation['misinformation']) == 0:
            if string:
                label = "unverified"
            else:
                label = 2
        elif int(annotation['misinformation']) == 1:
            if string:
                label = "false"
            else:
                label = 0

    elif ('true' in annotation.keys()) and (
          'misinformation' not in annotation.keys()):
        print ('Has true not misinformation')
        label = None
    else:
        print('No annotations')
        label = None
    return label


def tree2branches(root):
    node = root
    parent_tracker = []
    parent_tracker.append(root)
    branch = []
    branches = []
    i = 0
    siblings = None
    while True:
        node_name = node.keys()[i]
        branch.append(node_name)
        # get children of the node
        first_child = node.values()[i]
        # actually all chldren, all tree left under this node
        if first_child != []:  # if node has children
            node = first_child  # walk down
            parent_tracker.append(node)
            siblings = first_child.keys()
            i = 0  # index of a current node
        else:
            branches.append(deepcopy(branch))
            if siblings is not None:
                i = siblings.index(node_name)  # index of a current node
                while i+1 >= len(siblings):
                    # if the node does not have next siblings
                    if node is parent_tracker[0]:  # if it is a root node
                        return branches
                    del parent_tracker[-1]
                    del branch[-1]
                    node = parent_tracker[-1]      # walk up ... one step
                    node_name = branch[-1]
                    siblings = node.keys()
                    i = siblings.index(node_name)
                i = i+1    # ... walk right
    #            node =  parent_tracker[-1].values()[i]
                del branch[-1]
            else:
                return branches


# process tweet into features
def cleantweet(tweettext, tweet):
    if "media" in tweet["entities"]:
        for url in tweet["entities"]["media"]:
            tweettext = tweettext.replace(url["url"], "picpicpic")
    if "urls" in tweet["entities"]:
        for url in tweet["entities"]["urls"]:
            tweettext = tweettext.replace(url["url"], "urlurlurl")
    return tweettext


def str_to_wordlist(tweettext, tweet, remove_stopwords=False):
    # converts sentece to list of tokens/words
    #  Remove non-letters
    # NOTE: Is it helpful or not to remove non-letters?
    # str_text = re.sub("[^a-zA-Z]", " ", str_text)
    tweettext = cleantweet(tweettext, tweet)
    str_text = re.sub("[^a-zA-Z]", " ", tweettext)
    # Convert words to lower case and split them
    # words = str_text.lower().split()
    words = nltk.word_tokenize(str_text.lower())
    # Optionally remove stop words (false by default)
    # NOTE: generic list of stop words, should i remove them or not?
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    # 5. Return a list of words
    return(words)


def loadW2vModel():
    # Turn tweet into average of word vectors
    # LOAD PRETRAINED MODEL
    global model_GN
    print ("Loading the model")
    model_GN = gensim.models.KeyedVectors.load_word2vec_format(
                    'data/GoogleNews-vectors-negative300.bin', binary=True)
    print ("Done!")


def sumw2v(tweet, avg=True):
    global model_GN
    model = model_GN
    num_features = 300
    temp_rep = np.zeros(num_features)
    wordlist = str_to_wordlist(tweet['text'], tweet, remove_stopwords=False)
    for w in range(len(wordlist)):
        if wordlist[w] in model:
            temp_rep += model[wordlist[w]]
    if avg and len(wordlist) != 0:
        sumw2v = temp_rep/len(wordlist)
    else:
        sumw2v = temp_rep
    return sumw2v


def text_sumw2v(text, avg=True):
    global model_GN
    model = model_GN
    num_features = 300
    temp_rep = np.zeros(num_features)
    str_text = re.sub("[^a-zA-Z]", " ", text)
    wordlist = nltk.word_tokenize(str_text.lower())
    for w in range(len(wordlist)):
        if wordlist[w] in model:
            temp_rep += model[wordlist[w]]
    if avg and len(wordlist) != 0:
        sumw2v = temp_rep/len(wordlist)
    else:
        # sum
        sumw2v = temp_rep
    return sumw2v


def getW2vCosineSimilarity(words, wordssrc):
    global model_GN
    model = model_GN
    words2 = []
    for word in words:
        if word in model.wv.vocab:  # change to model.wv.vocab
            words2.append(word)
    wordssrc2 = []
    for word in wordssrc:
        if word in model.wv.vocab:  # change to model.wv.vocab
            wordssrc2.append(word)
    if len(words2) > 0 and len(wordssrc2) > 0:
        return model.n_similarity(words2, wordssrc2)
    return 0.


def tweet2features(tw, i, branch, conversation):
    tokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '',
                                tw['text'].lower()))
    srctweet = conversation['source']['text']
    prevtweet = []
    if i > 0:
        prevtweet_id = branch[i-1]
        if (i-1) == 0:
            prevtweet = srctweet
        else:
            for response in conversation['replies']:
                if prevtweet_id == response['id_str']:
                    prevtweet = response['text']
                    break
        if prevtweet == []:
            print ("I = ", i)
            print ("Conv id", conversation["id"])
            print ("Tweet id", tw["id"])
            print ("branch", branch)
        srctokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '',
                                              srctweet.lower()))
        prevtokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '',
                                               prevtweet.lower()))
    otherthreadtweets = ''
    if i != 0:
        otherthreadtweets += srctweet
    for response in conversation['replies']:
        if response['user']['screen_name'] != tw['user']['screen_name']:
            otherthreadtweets += ' ' + response['text']
    otherthreadtokens = nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '',
                                           otherthreadtweets.lower()))
    features = []
    tw['text'] = cleantweet(tw['text'], tw)
    issourcetw = int(tw['in_reply_to_screen_name'] is None)
    hasqmark = 0
    if tw['text'].find('?') >= 0:
        hasqmark = 1
    hasemark = 0
    if tw['text'].find('!') >= 0:
        hasemark = 1
    hasperiod = 0
    if tw['text'].find('.') >= 0:
        hasperiod = 0
    hasurl = 0
    if tw['text'].find('urlurlurl') >= 0 or tw['text'].find('http') >= 0:
        hasurl = 1
    haspic = 0
    if (tw['text'].find('picpicpic') >= 0) or (
                                  tw['text'].find('pic.twitter.com') >= 0) or (
                                  tw['text'].find('instagr.am') >= 0):
        haspic = 1
    hasnegation = 0
    negationwords = ['not', 'no', 'nobody', 'nothing', 'none', 'never',
                     'neither', 'nor', 'nowhere', 'hardly', 'scarcely',
                     'barely', 'don', 'isn', 'wasn', 'shouldn', 'wouldn',
                     'couldn', 'doesn']
    for negationword in negationwords:
        if negationword in tokens:
            hasnegation += 1
    charcount = len(tw['text'])
    wordcount = len(nltk.word_tokenize(re.sub(r'([^\s\w]|_)+', '',
                                              tw['text'].lower())))
    swearwords = []
    with open('badwords.txt', 'r') as f:
        for line in f:
            swearwords.append(line.strip().lower())
    hasswearwords = 0
    for token in tokens:
        if token in swearwords:
            hasswearwords += 1
    uppers = [l for l in tw['text'] if l.isupper()]
    capitalratio = len(uppers)/len(tw['text'])
    if i > 0:
        Word2VecSimilarityWrtSource = getW2vCosineSimilarity(tokens,
                                                             srctokens)
        Word2VecSimilarityWrtPrev = getW2vCosineSimilarity(tokens,
                                                           prevtokens)
    else:
        Word2VecSimilarityWrtSource = 0
        Word2VecSimilarityWrtPrev = 0
    Word2VecSimilarityWrtOther = getW2vCosineSimilarity(tokens,
                                                        otherthreadtokens)
    avgw2v = sumw2v(tw, avg=True)
    features = [charcount, wordcount, issourcetw, hasqmark, hasemark,
                hasperiod, hasurl, haspic, hasnegation, hasswearwords,
                capitalratio, Word2VecSimilarityWrtSource,
                Word2VecSimilarityWrtPrev, Word2VecSimilarityWrtOther]
    features.extend(deepcopy(avgw2v))
    features = np.asarray(features, dtype=np.float32)
    return features


def convert_label(label):
    if label == "true":
        return(0)
    elif label == "false":
        return(1)
    elif label == "unverified":
        return(2)
    else:
        print(label)
