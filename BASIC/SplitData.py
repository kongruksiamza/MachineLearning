from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris_dataset=load_iris()

x_t,x_s,y_t,y_s = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

print(x_t.shape)
print(x_s.shape)