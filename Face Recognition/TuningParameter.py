from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.svm import SVC

X,z = make_moons(n_samples=80,shuffle=0,noise=0.25,random_state=0)
mx,my = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),200),np.linspace(X[:,1].min(),X[:,1].max(),200))
mX = np.stack([mx.ravel(),my.ravel()],1)
plt.figure(figsize=[6.5,4.5])
for i,C in enumerate([1,5,10]):
    for j,gamma in enumerate([0.1,1,5]):
        svc = SVC(C=C,gamma=gamma)
        svc.fit(X,z)
        mz = svc.predict(mX).reshape(200,200)
        plt.subplot2grid((3,3),(i,j),xlim=[X[:,0].min(),X[:,0].max()],ylim=[X[:,1].min(),X[:,1].max()],xticks=[],yticks=[],aspect=1)
        plt.scatter(X[:,0],X[:,1],s=10,c=z,edgecolor='k',cmap='brg')
        plt.contourf(mx,my,mz,alpha=0.1,cmap='brg')
        plt.title('C=%.1f,$\\gamma$=%.1f'%(C,gamma),size=8)
plt.tight_layout()
plt.show()