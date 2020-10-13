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
