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

!pip install konlpy

from typing import List, Any
from collections import defaultdict
from konlpy.tag import Mecab


def duplicated(df: pd.DataFrame, target_col: str, n: int) -> dict:
    lst = []
    for title, sub in df.groupby('title'):
        if len(sub) >= n:
            res = set(sub[target_col])
            for value in res:
                lst.append(value)

    # {지역명 or 카테고리 : 횟수}
    tar_freq = defaultdict(int)
    for value in lst:
        tar_freq[value] += 1

    pairs = ((val, key) for (key, val) in tar_freq.items())
    sorted_pairs = sorted(pairs, reverse=True)
    final_res = {k: v for v, k in sorted_pairs}

    return final_res

from collections import defaultdict

def duplicated_loc(df:pd.DataFrame, N:int) -> dict:

    # 도배글이 많이 올라온 지역 : {'지역' : 횟수}
    loc_lst = [set(sub['location'].values)
                    for id, sub in df.groupby(['id', 'title']) if len(sub) >= 3]
    loc_lst = [val for loc in loc_lst for val in loc]
    loc_cnt = defaultdict(int)

    for loc in loc_lst:
        loc_cnt[loc] += 1

    # pairs = ((v, k) for (k, v) in loc_cnt.items())
    top_N = sorted(loc_cnt.items(), key = lambda x: x[1], reverse = True)[:N]
    return top_N
