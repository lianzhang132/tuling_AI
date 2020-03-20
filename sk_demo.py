from sklearn.datasets import load_iris

#加载数据
iris = load_iris()
x = iris.data
y = iris.target


#使用k临近分类器
#选择临近模型

#配置超参数n_neighbors = 1
from  sklearn.neighbors import KNeighborsClassifier



model = KNeighborsClassifier(n_neighbors=1)

#训练模型
model.fit(x,y)

#利用模型进行预测
y_model = model.predict(x)

#计算模型准确率
from sklearn.metrics import accuracy_score
rst = accuracy_score(y,y_model)

print("模型准确率：{}%".format(rst*100))
print(rst)