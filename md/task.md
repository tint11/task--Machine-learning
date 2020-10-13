csv文件数据格式：

- 第一列是样本名称
- 第2-513列是图片纹理特征
- 第514列是该样本的标签：1挖矿；0非挖矿

### task1

1. 使用t-SNE算法可视化所给样本的特征矩阵，并观察是否能通过纹理特征进行聚类

   结果：（可视化结果如下，大致可以通过纹理特征进行聚类，具体代码见图片下面）

![image-20201011113703341](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20201011113703341.png)

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_data():
    # 导入数据集
    df = open("bin_gist.csv")
    data = pd.read_csv(df)
    #print(type(data))
    # 自变量与因变量分离,自变量取第2列到513列，因变量为最后一列（0或1）
    x = data.iloc[:, 1:-1].values
    y = data.iloc[:, 513].values
    n_samples, n_features = data.shape
    return x, y, n_samples, n_features

def plot_embedding(data, label, title):   #画图
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 1),
                 fontdict={'weight': 'bold', 'size': 9})
    #设置坐标范围，刻度
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    my_x_ticks = np.arange(0, 1, 0.2)
    my_y_ticks = np.arange(0, 1, 0.2)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.title(title)
    return fig

def main():
    data, label, n_samples, n_features = get_data()
    #降维，降为二维
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits')
    plt.show(fig)

if __name__ == '__main__':
    main()
```

### task2

使用单模型对数据进行拟合（逻辑回归，knn，决策树，随机森林，SVM，朴素贝叶斯，梯度提升树，MLPClassifier等），具体要求如下：

- 使用sklearn库

- 五折交叉验证

- 使用网格搜索算法，搜索最优参数

- 使用acc，TP，TN，FP，FN，auc等参数评估分类器，并可视化比较结果

  结果：（导出的可视化结果保存为csv文件如下，具体代码见图片下面）

![image-20201013191612391](https://raw.githubusercontent.com/tint11/cloud-img/main/data/image-20201013191612391.png)

```
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier

def get_data():
    # 导入数据集
    df = open("bin_gist.csv")
    data = pd.read_csv(df)
    # 自变量与因变量分离,自变量取第2列到513列，因变量为最后一列（0或1）
    x = data.iloc[:, 1:-1].values
    y = data.iloc[:, 513].values
    # 随机从整个数据集中找到80%的数据作为训练集，另外20%的数据作为测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test

def Single_model():
    x_train, x_test, y_train, y_test = get_data()
    # 对数据进行特征缩放
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.fit_transform(x_test)
    # 选定LG, svm, tree, RF, GBDT, LGB单模型对数据进行拟合，提前设定参数变化值
    modle_list = ["LG", "svm", "tree", "RF", "GBDT", "LGB"]
    LG_modle = LogisticRegression(solver="liblinear")
    svm_modle = SVC(gamma='auto', probability=True)
    tree_modle = DecisionTreeClassifier()
    RF_modle = RandomForestClassifier()
    GBDT_modle = GradientBoostingClassifier()
    LGB_modle = LGBMClassifier()
    modle_dic = {"LG": LG_modle, "svm": svm_modle, "tree": tree_modle,
                 "RF": RF_modle, "GBDT": GBDT_modle, "LGB": LGB_modle}

    LG_param = {'penalty': ('l1', 'l2'), 'C': [0.1, 0.2, 0.5, 0.8, 1], "max_iter": [100, 150, 200]}
    svm_param = {}
    tree_param = {"splitter": ("best", "random"), "max_depth": [5, 10, 15, 20, 30, 50, 80],
                  "min_samples_leaf": [30, 50, 100]}
    RF_param = {"n_estimators": [100, 200, 500]}
    GBDT_param = {"learning_rate": [0.1, 0.2], "max_depth": [3, 5], "n_estimators": [100, 150, 200]}
    LGB_param = {"n_estimators": [100, 150, 200, 500]}
    modle_param = {"LG": LG_param, "svm": svm_param, "tree": tree_param,
                   "RF": RF_param, "GBDT": GBDT_param, "LGB": LGB_param}
    # 用数据帧(DataFrame)以行和列的表格方式排列指标和对应的模型，然后采用遍历，分别导出评估指标表保存为csv格式
    result = pd.DataFrame(columns=["LG", "svm", "tree", "RF", "GBDT", "LGB"],
                          index=["accuracy_score", "precision_score", "recall_score", "f1_score"])
    for name in modle_list:
        clf = GridSearchCV(modle_dic[name], modle_param[name], cv=5) # 网格搜索，cv=5表示五折交叉验证
        #进行数据拟合，通过训练得到模型参数
        clf.fit(x_train, y_train)
        print(clf.best_estimator_)
        model = clf.best_estimator_.fit(x_train, y_train)
        y_pre = model.predict(x_test)
        #得到最优参数模型后分别计算accuracy_score，precision_score(y_test, y_pre)，recall_score, f1_score，存到DataFrame类型的result中
        acc = metrics.accuracy_score(y_test, y_pre)
        pre = metrics.precision_score(y_test, y_pre, average='weighted') #使用weighted计算每个标签的指标，找到它们的平均加权权重（每个标签的真实实例数）
        rec = metrics.recall_score(y_test, y_pre, average='weighted')
        f1 = metrics.f1_score(y_test, y_pre, average='weighted')
        result[name] = [acc, pre, rec, f1]
    result.to_csv("D:/Code/task/result.csv")   #导出各模型的评估结果.csv文件

def main():
    Single_model()

if __name__ == '__main__':
    main()
```

### task3

使用Bagging、Boosting、Stack等模型框架对数据进行拟合（其基模型为第二步网格搜索得到的最优参数模型），观察集成学习相较于单模型学习的效果是否有所提升，可视化比较结果

结果：（导出的可视化结果分别导出为csv文件如下，由于GBDT模型运行不出来，已放弃，具体代码见图片下面）

![image-20201013192414334](https://raw.githubusercontent.com/tint11/cloud-img/main/data/image-20201013192414334.png)

![image-20201013192324300](https://raw.githubusercontent.com/tint11/cloud-img/main/data/image-20201013192324300.png)

![image-20201013192458356](https://raw.githubusercontent.com/tint11/cloud-img/main/data/image-20201013192458356.png)



```
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
def get_data():
    # 导入数据集
    df = open("bin_gist.csv")
    data = pd.read_csv(df)
    # 自变量与因变量分离,自变量取第2列到513列，因变量为最后一列（0或1）
    x = data.iloc[:, 1:-1].values
    y = data.iloc[:, 513].values
    # 随机从整个数据集中找到80%的数据作为训练集，另外20%的数据作为测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test

def search_the_best():
    x_train, x_test, y_train, y_test = get_data()
    # 对数据进行特征缩放
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.fit_transform(x_test)
    # 选定LG, svm, tree, RF, GBDT, LGB单模型对数据进行拟合，提前设定参数变化值
    modle_list = ["LG", "svm", "tree", "RF", "LGB"]
    LG_modle = LogisticRegression(solver="liblinear")
    svm_modle = SVC(gamma='auto', probability=True)
    tree_modle = DecisionTreeClassifier()
    RF_modle = RandomForestClassifier()
    # GBDT模型电脑一两个小时运行不出来，已放弃
   # GBDT_modle = GradientBoostingClassifier()
    LGB_modle = LGBMClassifier()
    modle_dic = {"LG": LG_modle, "svm": svm_modle, "tree": tree_modle,
                 "RF": RF_modle,  "LGB": LGB_modle}

    LG_param = {'penalty': ('l1', 'l2'), 'C': [0.1, 0.2, 0.5, 0.8, 1], "max_iter": [100, 150, 200]}
    svm_param = {}
    tree_param = {"splitter": ("best", "random"), "max_depth": [5, 10, 15, 20, 30, 50, 80],
                  "min_samples_leaf": [30, 50, 100]}
    RF_param = {"n_estimators": [100, 200, 500]}
    #GBDT_param = {"learning_rate": [0.1, 0.2], "max_depth": [3, 5], "n_estimators": [100, 150, 200]}
    LGB_param = {"n_estimators": [100, 150, 200, 500]}
    modle_param = {"LG": LG_param, "svm": svm_param, "tree": tree_param,
                   "RF": RF_param,  "LGB": LGB_param}
    model=[]
    for name in modle_list:
        print(name,'jjjjjj')
        clf = GridSearchCV(modle_dic[name], modle_param[name], cv=5)  # 网格搜索，cv=5表示五折交叉验证
        # 进行数据拟合，通过训练得到模型参数
        clf.fit(x_train, y_train)
        print(clf.best_estimator_)
        model.append(clf.best_estimator_.fit(x_train, y_train))
    return model, x_train, x_test, y_train, y_test

def the_best_to_integrated():
    modlelist, x_train, x_test, y_train, y_test = search_the_best()
    # 用数据帧(DataFrame)以行和列的表格方式排列指标和对应的模型，然后采用遍历，分别导出评估指标表保存为csv格式
    modle_list = ["LG", "svm", "tree", "RF", "LGB"]
    result = pd.DataFrame(columns=["LG", "svm", "tree", "RF", "LGB"],
                          index=["accuracy_score", "precision_score", "recall_score", "f1_score"])

    #使用Bagging分类器集成
    num=0
    for name in modle_list:

        print(name,'hhhh')
        clf1 = BaggingClassifier(base_estimator=modlelist[num], n_estimators=100, max_samples=1.0, max_features=1.0,
                                 bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1)
        clf1.fit(x_train, y_train)
        y_pre = clf1.predict(x_test)
        acc = metrics.accuracy_score(y_test, y_pre)
        pre = metrics.precision_score(y_test, y_pre, average='weighted')  # 使用weighted计算每个标签的指标，找到它们的平均加权权重（每个标签的真实实例数）
        rec = metrics.recall_score(y_test, y_pre, average='weighted')
        f1 = metrics.f1_score(y_test, y_pre, average='weighted')
        result[name] = [acc, pre, rec, f1]
        num=num+1
    result.to_csv("D:/Code/task/Bagging_result.csv")

    # 使用stacking分类器

    num = 0
    for name in modle_list:
        print(name, 'hhhh')
        estimators=[('rf', RandomForestClassifier(n_estimators=10, random_state=42)),('svr', make_pipeline(StandardScaler(),LinearSVC(random_state=42)))]
        clf2 = StackingClassifier(estimators=estimators, final_estimator=modlelist[num], cv=5, stack_method='auto', n_jobs=1, passthrough=False, verbose=0)
        clf2.fit(x_train, y_train)
        y_pre = clf2.predict(x_test)
        acc = metrics.accuracy_score(y_test, y_pre)
        pre = metrics.precision_score(y_test, y_pre, average='weighted')  # 使用weighted计算每个标签的指标，找到它们的平均加权权重（每个标签的真实实例数）
        rec = metrics.recall_score(y_test, y_pre, average='weighted')
        f1 = metrics.f1_score(y_test, y_pre, average='weighted')
        result[name] = [acc, pre, rec, f1]
        num = num + 1
    num = 0
    result.to_csv("D:/Code/task/stacking_result.csv")

    # 使用boosting分类器
    num=0
    for name in modle_list:

        clf3 = AdaBoostClassifier(base_estimator=modlelist[num], n_estimators=100, learning_rate=1.0, algorithm='SAMME.R',
                                  random_state=1)
        clf3.fit(x_train, y_train)
        y_pre = clf3.predict(x_test)
        acc = metrics.accuracy_score(y_test, y_pre)
        pre = metrics.precision_score(y_test, y_pre, average='weighted')  # 使用weighted计算每个标签的指标，找到它们的平均加权权重（每个标签的真实实例数）
        rec = metrics.recall_score(y_test, y_pre, average='weighted')
        f1 = metrics.f1_score(y_test, y_pre, average='weighted')
        result[name] = [acc, pre, rec, f1]
        num=num+1
    num = 0
    result.to_csv("D:/Code/task/boosting_result.csv")


def main():
    the_best_to_integrated()

if __name__ == '__main__':
    main()
```