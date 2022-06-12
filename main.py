# -*- coding: utf-8 -*-
# @Author : LuoXianan
# @File : main.py
# @Project: project_1
# @CreateTime : 2022/5/11 22:46:44

# 导入需要的类库
# 用pandas读取外部文件
import matplotlib.pyplot as plt
from pandas import read_csv
# 绘制散点图
from pandas.plotting import scatter_matrix
# 绘图
from matplotlib import pyplot
# sklearn分类需要的类
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# 交叉验证
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# 打分
from sklearn.metrics import accuracy_score
# 逻辑回顾算法
from sklearn.linear_model import LogisticRegression
# 决策树
from sklearn.tree import DecisionTreeClassifier
#
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# K近邻算法
from sklearn.neighbors import KNeighborsClassifier
# 贝叶斯
from sklearn.naive_bayes import GaussianNB
# 支持向量机SVM
from sklearn.svm import SVC


# # 导入iris数据集
# from sklearn.datasets import load_iris
# iris_dataset=load_iris()


# 导入数据，括号内为数据位置
filename='data.csv'
# names是给数据命名
names=['low_risk','medium_risk','high_risk','break_faith','class']
# 读入csv文件，使用pandas读入数据
dataset=read_csv(filename,names=names)

# 查看数据前五行
print('查看数据前五行:')
print(dataset.head())
print('-'*40)

# 查看数据维度
print('查看数据维度:')
print('数据维度：行 %s,列 %s' % dataset.shape)
print('-'*40)

# 统计描述数据信息
# 数据特征的统计描述信息包括数据的行数、中位数、最大值、最小值、均值、四分位值等统计数据信息
print('统计描述数据信息:')
print(dataset.describe())
print('-'*40)

# 数据分类分布情况
print('数据分类分布情况:')
print(dataset.groupby('class').size())
print('-'*40)

'''
常用的调整数据平衡的方法：
1.扩大数据样本；
2.数据的重新抽样；当数据超过一万条时，可以考虑测试欠抽样（删除多数类样本），当数据量比较少时可以考虑过抽样（复制少数类样本）；
3.尝试生成人工样本；
4.异常检测和变化检测
'''

# 数据可视化
'''
经过数据审查后，对数据有了一个基本的了解，
接下来用更直观的图标来进一步查看数据特征的分布情况。
使用单变量图表可以更好地理解每一个特征属性；
多变量图表用于理解不同特征属性之间的关系。
'''

# 1.单变量图表
# 单变量图表可以显示每一个单独的特征属性，
# 由于特征值都是数字，可以使用箱线图来表示属性与中位值的离散速度。
# 箱线图
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
pyplot.show()

# 绘制直方图
# 直方图
dataset.hist()
pyplot.show()

# 2.多变量图表
# 可以通过散点矩阵图来查看每个属性之间的关系
# 散点矩阵图
# 从多变量图大概能看出特征量之间的关系
scatter_matrix(dataset)
pyplot.show()

'''
4 评估算法
将数据集代入各种算法训练，找出最合适的算法。
步骤如下：
（1）分离训练集；
（2）采用10折交叉验证来评估算法模型；
（3）生成6个不同的模型来预测新数据；
（4）选择最优模型。
'''

# 1.分离训练集
# 一般分出数据集的80%作为训练集，剩下的20%用来作为测试集。
# 分出训练集
'''
这里的random_state就是为了保证程序每次运行都分割一样的训练集和测试集。
否则，同样的算法模型在不同的训练集和测试集上的效果不一样。
当你用sklearn分割完测试集和训练集，确定模型和初始参数以后，你会发现程序每运行一次，
都会得到不同的准确率，无法调参。这个时候就是因为没有加random_state。加上以后就可以调参了。

random_state是一个随机种子，是在任意带有随机性的类或函数里作为参数来控制随机模式
随机数种子控制每次划分训练集和测试集的模式，其取值不变时划分得到的结果一模一样，
其值改变时，划分得到的结果不同。若不设置此参数，则函数会自动选择一种随机模式，得到的结果也就不同。
'''
array=dataset.values
X=array[:,0:4]
Y=array[:,4]
validation_size=0.2  # 划分训练集的个数
seed=7
X_train,X_validation,Y_train,Y_validation=train_test_split(X,Y,test_size=validation_size,random_state=seed)
print('训练集为',X_train.shape)  # 花总数150个，训练集120个，测试集30个。分离成功

print('训练集的长度',len(X_train))
print('X_train为：',X_train)
print('X_validation为：',X_validation)
print('Y_train为',Y_train)
print('Y_validation为：',Y_validation)

print('-'*40)

# 2.评估模型
# 用10折交叉验证来分离训练数据集，评估算法的准确度。10折交叉验证是随机地将数据分成10份：9份用来训练模型，1份用来评估算法。


# 3.创建模型
'''
根据散点图可以看出，有些数据符合线性分许，所以可以用线性模型来评估。
用六种算法来评估：
线性回归（LR）；
线性判别分析（LDA）；
K近邻（KNN）；
分类与回归树（CART）；
贝叶斯分类器（NB）；
支持向量机（SVM）。
其中，LR和LDA为线性算法，剩下的都为非线性算法。
'''
#
# from sklearn.linear_model import LogisticRegression
# log_model = LogisticRegression(solver='lbfgs',max_iter=3000)
# model = LogisticRegression(max_iter=3000)
# model.fit(X_train,Y_train)

# 算法审查
models = {
        'LR' : LogisticRegression(max_iter=10000),  # 逻辑回归（LR）规定收敛次数为10000
        'LDA' : LinearDiscriminantAnalysis(),       # 线性判别分析（LDA）
        'KNN' : KNeighborsClassifier(),             # K近邻（KNN）
        'CART' : DecisionTreeClassifier(),          # 分类与回归树（CART）
        'NB' : GaussianNB(),                        # 贝叶斯分类器（NB）
        'SVM' : SVC(),                              # 支持向量机（SVM）
        'RFC': RandomForestClassifier()             # 随机森林分类(RFC)
        }
# 评估算法
results = []
n_splits=10
for key in models:
    kfold=KFold(n_splits=10,random_state=seed,shuffle=True)
    cv_results=cross_val_score(models[key],X_train,Y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    print('该算法准确率为:',key,sum(cv_results)/n_splits)
    # print('-' * 40)
    # print(key,results)
    # print('总和%s',sum(cv_results)/n_splits)
    # print('accuracy%s',sum(cv_results)/n_splits)
print('-'*40)


# 箱线图比较算法
fig=pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax=fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(models.keys())
pyplot.show()

'''
错误以及解决：
UndefinedMetricWarning:
解决：正如报错所说，你的模型的分类结果中有一类是没有被预测的，
拿2分类来说，你的模型全部预测成了1或者0，就会报上述错误。

我们的模型预测成1或者0关 sklearn 什么事呢？
这是因为如果某个类别没有被预测，Macro-F1会有除0的操作，所以才警告你一下。
'''
# 解决办法
import warnings
warnings.filterwarnings("ignore")



# 使用评估数据集评估算法
'''
 一般都是用准确率来作为评价指标，然而对于类别不均衡的任务来说，
 或者在任务中某一个类的准确率非常重要。如果再使用单纯的准确率肯定是不合理的，
 对任务来说 没有意义。所以我们需要一个好的评价指标来。
 目前一般都是用精准率，召回率，F1分数来评价模型；
在sklearn中有自动生成这些指标的的工具，
就是sklearn.metrics.classification_report模块
'''

# 实施预测
#  第一个算法
print('支持向量机SVM:')
svm=SVC()  # 支持向量机算法
svm.fit(X=X_train,y=Y_train)  # 匹配X训练集和Y训练集
predictions=svm.predict(X_validation)  # 预测值
# -----------------------------------------------------

# Y_validation类别的真实标签值 ,predictions预测值的标签
# print(accuracy_score(Y_validation,predictions))
print('1.The accuracy_score of our model is:',accuracy_score(Y_validation,predictions))
# ------------------------------------------------------
# confusion_matrix混淆矩阵
# 混淆矩阵是机器学习中总结分类模型预测结果的情形分析表，
# 以矩阵形式将数据集中的记录按照真实的类别与分类模型作出的分类判断两个标准进行汇总。
# 名字来源于它可以非常容易的表明多个类别是否有混淆（也就是一个class被预测成另一个class）
# 对角线为预测正确的值
print('2.confusion_matrix:')
print(confusion_matrix(Y_validation,predictions))
# ------------------------------------------------------
# 在深度学习中，分类任务评价指标是很重要的，一个好的评价指标对于训练一个好的模型极其关键；
# 如果评价指标不对，对于任务而言是没有意义的。
#  一般都是用准确率来作为评价指标，然而对于类别不均衡的任务来说，或者在任务中某一个类的准确率非常重要。
#  如果再使用单纯的准确率肯定是不合理的，对任务来说 没有意义。所以我们需要一个好的评价指标来。
#  目前一般都是用精准率，召回率，F1分数来评价模型；
#  在sklearn中有自动生成这些指标的的工具，就是  sklearn.metrics.classification_report模块
print('3.classification_report:')
print(classification_report(Y_validation,predictions))
print('-'*40)
'''
在这个报告中：
Y_validation 为样本真实标签，predictions 为样本预测标签；
support：当前行的类别在测试数据中的样本总量，如上表就是，在class 0 类别在测试集中总数量为1；
precision：精度=正确预测的个数(TP)/被预测正确的个数(TP+FP)；人话也就是模型预测的结果中有多少是预测正确的  查的准不准
recall:召回率=正确预测的个数(TP)/预测个数(TP+FN)；人话也就是某个类别测试集中的总量，有多少样本预测正确了； 查的全不全
f1-score:F1 = 2*精度*召回率/(精度+召回率)   反应模型的稳健性，越接近，模型的评估效果越好
micro avg：计算所有数据下的指标值，假设全部数据 5 个样本中有 3 个预测正确，所以 micro avg 为 3/5=0.6
macro avg：每个类别评估指标未加权的平均值，比如准确率的 macro avg，(0.50+0.00+1.00)/3=0.5
weighted avg：加权平均，就是测试集中样本量大的，我认为它更重要，给他设置的权重大点；比如第一个值的计算方法，(0.50*1 + 0.0*1 + 1.0*3)/5 = 0.70
'''

# 第二个算法
print('线性回归LR:')
LR = LogisticRegression(max_iter=10000)  # 线性回归
LR.fit(X=X_train,y=Y_train)
predictions=LR.predict(X_validation)
print('1.The accuracy_score of our model is:',accuracy_score(Y_validation,predictions))
print('2.confusion_matrix:')
print(confusion_matrix(Y_validation,predictions))
print('3.classification_report:')
print(classification_report(Y_validation,predictions))
print('-'*40)

# 第三个算法
print('贝叶斯分类器NB:')
NB = GaussianNB()               # 贝叶斯算法
NB.fit(X=X_train,y=Y_train)
predictions=NB.predict(X_validation)
print('1.The accuracy_score of our model is:',accuracy_score(Y_validation,predictions))
print('2.confusion_matrix:')
print(confusion_matrix(Y_validation,predictions))
print('3.classification_report:')
print(classification_report(Y_validation,predictions))
print('-'*40)

# 第四个算法
print('线性判别分析LDA:')
LDA = LinearDiscriminantAnalysis()   #
LDA.fit(X=X_train,y=Y_train)
predictions=LDA.predict(X_validation)
print('1.The accuracy_score of our model is:%s',accuracy_score(Y_validation,predictions))
print('2.confusion_matrix:')
print(confusion_matrix(Y_validation,predictions))
print('3.classification_report:')
print(classification_report(Y_validation,predictions))
print('-'*40)

# 第五个算法
print('K近邻KNN:')
KNN = KNeighborsClassifier()
KNN.fit(X=X_train,y=Y_train)
predictions=KNN.predict(X_validation)
print('1.The accuracy_score of our model is:',accuracy_score(Y_validation,predictions))
print('2.confusion_matrix:')
print(confusion_matrix(Y_validation,predictions))
print('3.classification_report:')
print(classification_report(Y_validation,predictions))
print('-'*40)

# 第六个算法
print('分类与回归树CART')
CART = DecisionTreeClassifier()
CART.fit(X=X_train,y=Y_train)
predictions=CART.predict(X_validation)
print('1.The accuracy_score of our model is:',accuracy_score(Y_validation,predictions))
print('2.confusion_matrix:')
print(confusion_matrix(Y_validation,predictions))
print('3.classification_report:')
print(classification_report(Y_validation,predictions))
print('-'*40)

# 第七个算法
# 'RFC': RandomForestClassifier()
print('随机森林分类RFC')
RFC = RandomForestClassifier()
RFC.fit(X=X_train,y=Y_train)
predictions=CART.predict(X_validation)
print('1.The accuracy_score of our model is:',accuracy_score(Y_validation,predictions))
print('2.confusion_matrix:')
print(confusion_matrix(Y_validation,predictions))
print('3.classification_report:')
print(classification_report(Y_validation,predictions))
print('-'*40)