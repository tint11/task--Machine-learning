from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
#导入数据
iris_data = load_iris()
# X_train,X_test,y_train,y_test = train_test_split(iris_data.data,iris_data.target,random_state=0)
X_trainval, X_test, y_trainval, y_test = train_test_split(iris_data.data, iris_data.target, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=1)
# grid search start
best_score = 0
for gamma in [0.001, 0.01, 1, 10, 100]:
    for c in [0.001, 0.01, 1, 10, 100]:
        # 对于每种参数可能的组合，进行一次训练
        svm = SVC(gamma=gamma, C=c)
        # 5 折交叉验证
        scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
        score = scores.mean()
        # 找到表现最好的参数
        if score > best_score:
            best_score = score
            best_parameters = {'gamma': gamma, "C": c}

# 使用最佳参数，构建新的模型
svm = SVC(**best_parameters)

# 使用训练集和验证集进行训练 more data always resultd in good performance
svm.fit(X_trainval, y_trainval)

# evalyation 模型评估
test_score = svm.score(X_test, y_test)

print('Best socre:{:.2f}'.format(best_score))
print('Best parameters:{}'.format(best_parameters))
print('Best score on test set:{:.2f}'.format(test_score))