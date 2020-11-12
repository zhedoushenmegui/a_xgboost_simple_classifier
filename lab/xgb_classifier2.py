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
from sklearn import metrics
from sklearn.model_selection import train_test_split
import arrow
import time
import json
import traceback
import xgboost


class XgbClassifier2:
    def __init__(self, tnf, target_key, ts_ratio=.3, feats=None, model_file=None, save_folder=None):
        """
        :param tnf: pd.DataFrame 训练集
        :param target_key: str label的名字
        :param ts_ratio: float 测试集比例
        :param feats: list 特征的数组
        :param save_folder: str 模型和其他信息文件的路径, 可以从这里加载继续训练
        """
        self.general_params = {
            'booster': 'gbtree',  # gbtree, gblinear, dart
            'silent': 1,  #
            'verbosity': 0,  # 0 (silent), 1 (warning), 2 (info), 3 (debug)
            'nthread': 2,
            'disable_default_eval_metric': 0,  # Set to >0 to disable.
        }
        self.tree_params = {
            'eta': 0.1,  # learning_rate, default 0.3
            'gamma': 0.1,  # min_split_loss  越大越保守
            'max_depth': 5,
            'min_child_weight': 1,  # 越大越保守, 叶子节点上所有样本的权重和小于min_child_weight则停止分裂
            'max_delta_step': 1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'colsample_bylevel': 1,
            'colsample_bynode': 1,
            'lambda': 1.,  # L2
            'alpha': 0.1,
            # 'gpu_id': 0,
            'tree_method': 'hist',  # [auto, exact, approx, hist, gpu_hist]
            'grow_policy': 'lossguide',  # [depthwise, lossguide] only if tree_method is set to hist
            'max_leaves': 0,  # Only relevant when grow_policy=lossguide is set
            'max_bin': 256,  # Only used if tree_method is set to hist. 直方图数目, 越大越容易过拟合
        }
        self.task_params = {
            'objective': 'binary:logistic',  #
            'eval_metric': 'error',  # logloss, error, merror, mlogloss, auc, aucpr,
        }
        ###
        self.data = tnf.copy()
        self.feats = feats[:] if feats is not None else list(self.data.columns)
        if target_key in self.feats:
            self.feats.remove(target_key)
        ###
        self.target_key = target_key
        self.ori_feats = self.feats[:]
        self.feats = [f'f{i}' for i in range(len(self.feats))]
        self.feats_map0 = {a: b for a, b in zip(self.ori_feats, self.feats)}
        self.feats_map = {b: a for a, b in zip(self.ori_feats, self.feats)}
        self.data.rename(columns=self.feats_map0, inplace=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.data[self.target_key], test_size=ts_ratio, random_state=10)
        feats = self.feats
        self.d_train = xgboost.DMatrix(self.X_train[feats], label=self.y_train, feature_names=self.ori_feats)
        self.d_test = xgboost.DMatrix(self.X_test[feats], label=self.y_test, feature_names=self.ori_feats)
        ###
        self.bst = None if model_file is None else xgboost.Booster(model_file=model_file)
        ###
        self.run_round = 0
        ###
        self.eval_rst = {'tn': {}, 'ts': {}}
        self.preds = None
        ###
        if save_folder is not None:
            self.load_from_file(save_folder)

    def train(self, num_boost_round=5000, ifeval=0, ifcontinue=True):
        ts0 = time.time()
        ###
        try:
            xgb_params = {**self.general_params, **self.tree_params, **self.task_params}
            if ifeval:
                evals = [(self.d_train, 'train'), (self.d_test, 'test')]
                if ifcontinue and self.bst is not None:
                    self.bst = xgboost.train(xgb_params, self.d_train, num_boost_round=num_boost_round, evals=evals,
                                             xgb_model=self.bst)
                else:
                    self.bst = xgboost.train(xgb_params, self.d_train, num_boost_round=num_boost_round, evals=evals)
            else:
                if ifcontinue and self.bst is not None:
                    self.bst = xgboost.train(xgb_params, self.d_train, num_boost_round=num_boost_round, xgb_model=self.bst)
                else:
                    self.bst = xgboost.train(xgb_params, self.d_train, num_boost_round=num_boost_round)

            self.run_round = num_boost_round + (self.run_round if ifcontinue else 0)
            print("> 测试集")
            preds1 = self.bst.predict(self.d_test)
            self.eval_rst['ts'] = self.estimate(self.y_test, preds1)
            ###
            print("> 训练集")
            preds2 = self.bst.predict(self.d_train)
            self.eval_rst['tn'] = self.estimate(self.y_train, preds2)
            self.preds = list(preds1) + list(preds2)
        except:
            print(traceback.format_exc())

        ts1 = time.time()
        print("> train and estimate duration: {}, round: {}".format(round(ts1 - ts0, 2), self.run_round))

    def predict(self, df):
        df1 = df.rename(columns=self.feats_map0)
        dm = xgboost.DMatrix(df1[self.feats])
        return self.bst.predict(dm)

    @staticmethod
    def estimate(arr1, preds):
        arr2 = [1 if x > 0.5 else 0 for x in preds]

        n = len(arr1)
        t = sum([1 if a == b else 0 for a, b in zip(arr1, arr2)])
        acc = round(t / n, 5) if n else None
        print("> Accuracy: {}".format(acc))

        m = 0
        r = 0
        for a, b in zip(arr1, arr2):
            if a == 1:
                m += 1
                if b == 1:
                    r += 1
        recall_ = round(r / m, 4) if m else None
        print("> Recall: {}".format(recall_))

        m = 0
        p = 0
        for a, b in zip(arr1, arr2):
            if b == 1:
                m += 1
                if a == 1:
                    p += 1
        precision_ = round(p / m, 5) if m else None
        print("> Precision: {}".format(precision_))
        rst = {
            'recall': recall_,
            'precision': precision_,
            'acc': acc,
        }
        try:
            fpr, tpr, thresholds = metrics.roc_curve(arr1, preds, pos_label=1)
            auc = round(metrics.auc(fpr, tpr), 4)
            rst['auc'] = auc
            print("> Auc: {}".format(auc))
        except:
            pass
        return rst

    def _info(self, key='gain', ori=True):
        arr = sorted(self.bst.get_score(importance_type=key).items(), key=lambda x: x[1], reverse=True)
        if not ori:
            return arr
        else:
            return [(self.feats_map.get(x[0], x[0]), x[1]) for x in arr]

    def gain_info(self, ori=True):
        return self._info(key='gain', ori=ori)

    def weight_info(self, ori=True):
        return self._info(key='weight', ori=ori)

    def cover_info(self, ori=True):
        return self._info(key='cover', ori=ori)

    def dump_model(self, file_path):
        """model  txt"""
        self.bst.dump_model(file_path, with_stats=True)

    def save_model(self, model_path):
        """二进制模型"""
        self.bst.save_model(model_path)

    def dump_everything(self, folder, sample_size=200):
        """
        :param folder: str 保存路径
        :param sample_size: int 对数据采样, 用于判断python 和pmml 预测是否一致
        """
        print("> folder: {}".format(folder))
        os.system(f"mkdir -p {folder}")
        ### fmap
        with open(f'{folder}/fmap.txt', 'w') as f:
            f.write('\n'.join([f'{i}\t{x}\tq' for i, x in enumerate(self.feats)]))
        print("> fmp done!")
        ### gain
        with open(f'{folder}/gain.txt', 'w') as f:
            f.write("\n".join([f"{x[0]}: {x[1]}" for x in self.gain_info()]))
        print("> gain done!")
        ### cover
        with open(f'{folder}/cover.txt', 'w') as f:
            f.write("\n".join([f"{x[0]}: {x[1]}" for x in self.cover_info()]))
        print("> cover done!")
        ### weight
        with open(f'{folder}/weight.txt', 'w') as f:
            f.write("\n".join([f"{x[0]}: {x[1]}" for x in self.weight_info()]))
        print("> wight done!")
        ### model
        self.dump_model(f'{folder}/model.txt')
        print("> dump model done!")
        self.save_model(f'{folder}/xgb.model')
        print("> save model done!")
        ###
        cmd = f'java -jar {project_path}/common/jpmml-xgboost-executable-1.4-SNAPSHOT.jar --model-input '\
              f'{folder}/xgb.model --fmap-input {folder}/fmap.txt --target-name {self.target_key} --pmml-output '\
              f'{folder}/xgb.pmml'
        os.system(cmd)
        print("> convert model to pmml done!")
        ###
        data = {
            'general_params': self.general_params,
            'tree_params': self.tree_params,
            'task_params': self.task_params,
            'feats_map': self.feats_map,
            'feats_map0': self.feats_map0,
            'run_round': self.run_round,
            'eval_tn': self.eval_rst['tn'],
            'eval_ts': self.eval_rst['ts'],
            'save_time': arrow.now().format('YYYY-MM-DD HH:mm:ss'),
            'target_key': self.target_key,
        }
        with open(f'{folder}/info.json', 'w') as f:
            json.dump(data, f)
        print("> info done!")
        ###
        if self.preds is not None:
            with open(f'{folder}/preds.csv', 'w') as f:
                f.write('\n'.join([str(x) for x in self.preds]))
        ###
        sampledf = self.data.sample(200).copy()
        preds = self.bst.predict(xgboost.DMatrix(sampledf[self.feats], feature_names=self.ori_feats))
        keys = list(self.data.columns)
        arr = sampledf.apply(lambda row: json.dumps({k: x for k, x in zip(keys, row)}), axis=1)
        with open(f'{folder}/sample.txt', 'w') as f:
            for v,line in zip(preds, arr):
                f.write("%.5f#%s \n" % (v, line))

    def load_from_file(self, folder):
        if not os.path.exists(folder):
            print(f"{folder} not exist!")
            return
        if not os.path.exists(folder + '/' + 'xgb.model'):
            print(f"xgb.model not exist")
        else:
            self.bst = xgboost.Booster(model_file=folder + '/' + 'xgb.model')
        if not os.path.exists(folder + '/' + 'info.json'):
            print(f"info.json not exist")
        else:
            with open(f'{folder}/info.json', 'r') as f:
                cnt = f.read()
                obj = json.loads(cnt)
                self.general_params = obj.get('general_params')
                self.tree_params = obj.get('tree_params')
                self.task_params = obj.get('task_params')
                self.feats_map = obj.get('feats_map')
                self.feats_map0 = obj.get('feats_map0')
                self.run_round = obj.get('run_round')


def main():
    pass


if __name__ == '__main__':
    pass