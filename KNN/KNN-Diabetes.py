from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Read Data
df=pd.read_csv("diabetes.csv")
# data
x=df.drop("Outcome",axis=1).values
# outcome data
y=df['Outcome'].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)

knn=KNeighborsClassifier(n_neighbors=8)
#train
knn.fit(x_train,y_train)

#prediction
y_pred=knn.predict(x_test)

print(pd.crosstab(y_test,y_pred,rownames=['Actually'],colnames=['Prediction'],margins=True))

