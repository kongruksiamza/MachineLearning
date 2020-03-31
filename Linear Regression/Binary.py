from scipy.io import loadmat
import numpy as np
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt  

def displayImage(x):
    plt.imshow(x.reshape(28,28),cmap=plt.cm.binary,interpolation="nearest")
    plt.show()

def displayPredict(clf,actually_y,x):
    print("Actual = " ,actually_y)
    print("Predict = " , clf.predict([x])[0])


mnist_raw=loadmat("mnist-original.mat")
mnist={
    "data":mnist_raw["data"].T,
    "target":mnist_raw["label"][0]
}

x,y=mnist['data'],mnist['target']
x_train,x_test,y_train,y_test=x[:60000],x[60000:],y[:60000],y[60000:]


y_train_0=(y_train==0)

sgd=SGDClassifier()
sgd.fit(x_train,y_train_0)

predict_data = 5500

displayImage(x_test[5500])
displayPredict(sgd,y_train_0[predict_data],x_test[predict_data])






