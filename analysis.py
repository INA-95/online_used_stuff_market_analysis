!pip install datasets

from datasets import load_dataset
DOWNLOAD = "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"
dataset = load_dataset("psyche/daangn", use_auth_token=DOWNLOAD)

!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf

import os
import tensorflow as tf

cwd = os.getcwd()

path_mecab_zip = tf.keras.utils.get_file(
    'mecab-0.996-ko-0.9.2.tar.gz', origin='https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz',
    extract=True)

path_mecab_dic_zip = tf.keras.utils.get_file(
    'mecab-ko-dic-2.1.1-20180720.tar.gz', origin='https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz',
    extract=True)

os.chdir(os.path.join(os.path.dirname(path_mecab_zip),'mecab-0.996-ko-0.9.2/'))
!./configure
!make
!make check
!sudo make install

os.chdir(os.path.join(os.path.dirname(path_mecab_zip), 'mecab-ko-dic-2.1.1-20180720/'))
!sudo ldconfig
!ldconfig -p | grep /usr/local
!./configure
!make
!sudo make install

!pip install mecab-python3
!apt-get update
!apt-get install g++ openjdk-8-jdk python-dev python3
!pip3 install JPype1-py3
!pip3 install konlpy
!JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
%cd {cwd}

import pandas as pd

val_df = pd.DataFrame(dataset['validation'])
val_df.head()