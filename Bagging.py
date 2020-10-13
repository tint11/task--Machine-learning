from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.ensemble import BaggingClassifier

from sklearn.svm import SVC

#from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data['data'], data['target'], train_size=0.8, random_state=2020)


"""
data=[]
traffic_feature=[]
traffic_target=[]
csv_file = csv.reader(open('packSize_all.csv'))
for content in csv_file:
    content=list(map(float,content))
    if len(content)!=0:
        data.append(content)
        traffic_feature.append(content[0:6])#存放数据集的特征
        traffic_target.append(content[-1])#存放数据集的标签
print('data=',data)
print('traffic_feature=',traffic_feature)
print('traffic_target=',traffic_target)
scaler = StandardScaler() # 标准化转换
scaler.fit(traffic_feature)  # 训练标准化对象
traffic_feature= scaler.transform(traffic_feature)   # 转换数据集
feature_train, feature_test, target_train, target_test = train_test_split(traffic_feature, traffic_target, test_size=0.3,random_state=0)
tree=DecisionTreeClassifier(criterion='entropy', max_depth=None)
# n_estimators=500:生成500个决策树
"""
clf = BaggingClassifier(base_estimator=svm, n_estimators=500, random_state=0)
clf.fit(X_train,y_train)
predict_results=clf.predict(X_test)
print(accuracy_score(predict_results, y_test))
conf_mat = confusion_matrix(y_test, predict_results)
print(conf_mat)
print(classification_report(y_test, predict_results))