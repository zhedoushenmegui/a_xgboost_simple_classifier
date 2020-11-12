#!/usr/bin/env python
"""
author: lemon
create date: 2020/11/8
description:
history:
2020/11/8    lemon    init
"""

import sys
import os


project_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.'))
sys.path.append(project_path)
from xgbClassifier2 import XgbClassifier2
import pandas as pd


def main():
    # load data
    df = pd.read_csv(f"{project_path}/data/train.csv")
    print(df.shape)
    feats = list(df.columns)
    # id 没有意义, left 是label, sales和salary 需要处理
    for f in ['id', 'left', 'sales', 'salary']:
        if f in feats:
            feats.remove(f)
    #
    xgbc = XgbClassifier2(tnf=df, target_key='left', ts_ratio=.3, feats=feats)
    xgbc.train(num_boost_round=50)

    print(f'> everything is saved : {project_path}/output/sample')
    xgbc.dump_everything(f"{project_path}/output/sample")   



if __name__ == "__main__":
    main()