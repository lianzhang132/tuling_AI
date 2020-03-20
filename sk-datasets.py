from sklearn.datasets import *
rst = get_data_home()
#默认数据集的存放路径
print(rst)

#鸢尾花iris问题
from sklearn.datasets import load_iris

#载入数据
iris_data = load_iris()

#花败花萼数据
print("features Names:{} \n".format(iris_data.feature_names))
print("\nthe first 5 target: \n {}".format(iris_data.data[:5]))

#结果数据
print("\ntargetname:\n {}".format(iris_data.target_names))
print("\nthe first 5 target: \n {}".format(iris_data.target[::10]))

#数据的键值
print("\n all keys:\n {}" .format(iris_data.keys()))