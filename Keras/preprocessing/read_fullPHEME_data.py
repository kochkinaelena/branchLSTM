"""
This code contains functions for loading fullPHEME dataset
"""
import os
import json
from copy import deepcopy
import help_prep_functions
#%%


def read_fullPHEME():
    rnrfolds = ['germanwings-crash', 'ferguson', 'charliehebdo',
                'ottawashooting', 'sydneysiege', 'putinmissing',
                'prince-toronto', 'ebola-essien', 'gurlitt']
    rnr_annotations = []
    path_to_rnr_folds = 'all-rnr-annotated-threads'
    conversation = {}
    noreplyconv = 0
    preprocessed_data = {}
    cvfolds = {}
    for fold in rnrfolds:
        preprocessed_data[fold+'-all-rnr-threads'] = []
        cvfolds[fold] = []
        rnr_annotations = []
        path_to_rumours = os.path.join(path_to_rnr_folds,
                                       fold+'-all-rnr-threads',
                                       'rumours')
        rnrthreads = os.listdir(path_to_rumours)
        newfolds = [i for i in rnrthreads if i[0] != '.']
        rnrthreads = newfolds
        for thread in rnrthreads:
            path_to_source = os.path.join(path_to_rumours,
                                          thread,
                                          'source-tweets')
            src_tw_folder = os.listdir(path_to_source)
            newfolds = [i for i in src_tw_folder if i[0] != '.']
            src_tw_folder = newfolds
            path_to_source_tw = os.path.join(path_to_source, src_tw_folder[0])
            with open(path_to_source_tw) as f:
                    for line in f:
                        src = json.loads(line)
            # FILTER OUT GERMAN TWEETS: if source tweet lang is eng then keep
            if src['lang'] == 'en':
                path_struct = os.path.join(path_to_rumours,
                                           thread,
                                           'structure.json')
                with open(path_struct) as f:
                        for line in f:
                            struct = json.loads(line)
                # JUST GETTING RID of conversations which do not have structure
                if struct != []:
                    if len(struct) > 1:
                        # print "Structure has more than one root",
                        if thread in struct.keys():
                            new_struct = {}
                            new_struct[thread] = struct[thread]
                            struct = new_struct
                        else:
                            new_struct = {}
                            new_struct[thread] = struct
                            struct = new_struct
                    conversation['structure'] = struct
                    path_to_rnr_annotation = os.path.join(path_to_rumours,
                                                          thread,
                                                          'annotation.json')
                    with open(path_to_rnr_annotation) as f:
                        for line in f:
                            an = json.loads(line)
                            an['id'] = thread
                            rnr_annotations.append(an)
                    conversation['id'] = thread
                    conversation['veracity'] = help_prep_functions.convert_annotations(
                                                an,
                                                string=True)
                    conversation['source'] = src
                    tweets = []
                    path_repl = os.path.join(path_to_rumours,
                                             thread,
                                             'reactions')
                    files_t = os.listdir(path_repl)
                    if len(files_t) < 1:
                        # print "No replies", thread
                        noreplyconv = noreplyconv+1
                    newfolds = [i for i in files_t if i[0] != '.']
                    files_t = newfolds
                    for repl_file in files_t:
                        with open(os.path.join(path_repl, repl_file)) as f:
                            for line in f:
                                tw = json.loads(line)
                                tw['used'] = 0
                                tw['label'] = conversation['veracity']
                                tw['conv_id'] = thread
                                tweets.append(tw)
                                if tw['text'] is None:
                                    print ("Tweet has no text", tw['id'])
                    conversation['replies'] = tweets
                    cvfolds[fold].append(deepcopy(conversation))
    return cvfolds
