from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#load dataset
iris=load_iris()
#assign attribute , target
x=iris['data']
y=iris['target']

# train , test 
x_train,x_test,y_train,y_test=train_test_split(x,y)

#model
model=GaussianNB()
#train
model.fit(x_train,y_train)

#prediction
y_pred=model.predict(x_test)

#Accuracy Score
print("Accuracy = ",accuracy_score(y_test,y_pred))
