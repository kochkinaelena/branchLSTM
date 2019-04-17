import os
import wget
import subprocess


# change into downloaded_data folder
os.chdir("downloaded_data")

# download the datasets
f1 = wget.download(
    "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
)

f2 = wget.download(
    "https://s3-eu-west-1.amazonaws.com/downloads.gate.ac.uk/pheme/semeval2017-task8-dataset.tar.bz2"
)

f3 = wget.download(
    "http://alt.qcri.org/semeval2017/task8/data/uploads/rumoureval2017-test.tar.bz2"
)

# extract the datasets
cmd1 = ["gzip", "-d", f1]
subprocess.call(cmd1)

cmd2 = ["tar", "-xf", f2]
subprocess.call(cmd2)

cmd3 = ["tar", "-xf", f3]
subprocess.call(cmd3)

# Tidy up
os.remove("semeval2017-task8-dataset.tar.bz2")
os.remove("rumoureval2017-test.tar.bz2")
