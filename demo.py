# -*- coding: utf-8 -*-
# @Author : LuoXianan
# @File : demo.py
# @Project: project_1
# @CreateTime : 2022/5/11 22:45:13

#  对安装的库进行版本检测
import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
import IPython
print('python version:{}'.format(sys.version))
print('scipy version:{}'.format(scipy.__version__))
print('numpy version:{}'.format(numpy.__version__))
print('matplotlib version:{}'.format(matplotlib.__version__))
print('pandas version:{}'.format(pandas.__version__))
print('sklearn version:{}'.format(sklearn.__version__))
print('IPython version:{}'.format(IPython.__version__))