=======================================================
Dataset of social media rumours for SemEval 2017 task 8
=======================================================

This directory contains the train and development sets of the rumour dataset for SemEval 2017 Task 8. These rumours are associated with 9 different breaking news. It contains Twitter conversations which are initiated by a rumourous tweet; the conversations include tweets responding to those rumourous tweets. These tweets have been annotated for support, deny, query or comment (SDQC).

The code we used for the collection of conversations from Twitter is available on GitHub: https://github.com/azubiaga/pheme-twitter-conversation-collection

Data Structure
==============

The dataset contains 297 conversational threads, with a folder for each thread, and structured as follows:

 * source-tweet: This folder contains a json file with the source tweet.

 * replies: This folder contains the json files for all the tweets that participated in the conversations by replying.

 * structure.json: This file provides the structure of the conversation, making it easier to determine what each tweets children tweets are and to reconstruct the conversations by putting together the source tweet and the replies.

 * urls.dat: This file contains information about the URLs mentioned in the tweets. The file three tab-separated columns: md5 hash of the URL (which is used an ID of the URL), the URL as it appears in the tweet, and the long URL.

 * context: This folder contains context files where available. These include a Wikipedia file with the revision of the related Wikipedia article just prior to the time the source tweet was posted, as well as URL files also in their revision prior to being mentioned in the tweets, which are named according to the MD5 hashes in the urls.dat files.

Acknowledgment:
===============

The development of this dataset has been supported by the PHEME FP7 project (grant No. 611233).
