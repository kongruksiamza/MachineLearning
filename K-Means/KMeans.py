from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

x,y=make_blobs(n_samples=300,centers=4,cluster_std=0.5,random_state=0)

#new point
x_test,y_test=make_blobs(n_samples=10,centers=4,cluster_std=0.5,random_state=0)

# print(x[:,1])
# print(y.shape)
model=KMeans(n_clusters=4)
model.fit(x)
y_pred=model.predict(x)
y_pred_new=model.predict(x_test)
centers=model.cluster_centers_

# print(centers)
plt.scatter(x[:,0],x[:,1],c=y_pred)
plt.scatter(x_test[:,0],x_test[:,1],c=y_pred_new,s=120)
plt.scatter(centers[0,0],centers[0,1],c='blue',label="Centroid 1")
plt.scatter(centers[1,0],centers[1,1],c='green',label="Centroid 2")
plt.scatter(centers[2,0],centers[2,1],c='red',label="Centroid 3")
plt.scatter(centers[3,0],centers[3,1],c='black',label="Centroid 4")
plt.legend(frameon=True)
plt.show()