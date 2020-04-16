from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.svm import SVC

X,z = make_blobs(n_samples=100,n_features=2,cluster_std=4,centers=2,random_state=3)
mx,my = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),200),np.linspace(X[:,1].min(),X[:,1].max(),200))
mX = np.stack([mx.ravel(),my.ravel()],1)
plt.figure(figsize=[6,7])
kernel = ['rbf','poly','sigmoid','linear']
for i in range(4):
    svc = SVC(kernel=kernel[i])
    svc.fit(X,z)
    mz = svc.predict(mX).reshape(200,200)
    plt.subplot(2,2,i+1,aspect=1,xlim=[X[:,0].min(),X[:,0].max()],ylim=[X[:,1].min(),X[:,1].max()])
    plt.scatter(X[:,0],X[:,1],s=50,c=z,edgecolor='k',cmap='brg')
    plt.contourf(mx,my,mz,alpha=0.1,cmap='brg')
    plt.title(kernel[i])
plt.tight_layout()
plt.show()