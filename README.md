# a xgboost simple classifier
## 简介
- lab 下面是数据和python 脚本, 训练模型在这里
- lab/output 保存模型相关信息, xgbClassifier2.py 可以从加载上次保存的数据, 继续训练
- 使用 [jpmml](https://github.com/jpmml/jpmml-xgboost) 把模型转成pmml 文件, 我打了一个jar在lab/common
- src/main/java/dev/PmmlRunClassifier.java 是java 加载模型和预测的, 使用spring boot 简单包装就可以做一个服务

## 其他
python : 3.7+<br>
java : 8+<br>
lab里的训练仅仅为了演示, 特征处理, 仅仅训练了50轮<br>