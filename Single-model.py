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