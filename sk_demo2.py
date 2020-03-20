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

# print("模型准确率：{}%".format(rst*100))
# print(rst)

from sklearn.model_selection import train_test_split
#将训练集和测试集各占50%
x1,x2,y1,y2 = train_test_split(x,y,random_state=0,train_size=0.5)

#使用一部分数据训练
model.fit(x1,y1)

#使用另一部分测试

y2_model = model.predict(x2)

#比对
rst2 = accuracy_score(y2,y2_model)

print("模型准确率：{}%".format(rst2*100))
y2_model = model.fit(x1, y1).predict(x2)
y1_model = model.fit(x2, y2).predict(x1)

rst1 = accuracy_score(y1, y1_model)
rst2 = accuracy_score(y2, y2_model)

print("模型1准确率：{}％".format(rst1 * 100))
print("模型2准确率：{}％".format(rst2 * 100))
from sklearn.model_selection import cross_val_score
rst = cross_val_score(model, x, y, cv=5)
print("模型准确率：{}".format(rst * 100))

from sklearn.model_selection import LeaveOneOut, cross_val_score

scores = cross_val_score(model, x, y, cv=LeaveOneOut())

print("验证分数总共：{}".format(scores))

# 每次结果求均值作为最终结果
print("\n最终验证结果：{}".format(scores.mean()))