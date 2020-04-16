from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
faces=fetch_lfw_people(min_faces_per_person=60)

# print(faces.target_names)
# print(faces.images.shape)
# fig,ax=plt.subplots(3,5)
# for i,axi in enumerate(ax.flat):
#     axi.imshow(faces.images[i],cmap='bone')
#     axi.set(xticks=[],yticks=[])
#     axi.set_ylabel(faces.target_names[faces.target[i]].split()[-1],color='black')
# plt.show()

pca=PCA(n_components=150,svd_solver='randomized', whiten=True)
svc=SVC(kernel="rbf",class_weight="balanced")
model=make_pipeline(pca,svc)
x_train,x_test,y_train,y_test=train_test_split(faces.data,faces.target,random_state=45)

#parameter
param_grid={'svc__C':[1,5,10,50],'svc__gamma':[0.0001,0.005,0.001,0.005]}
grid=GridSearchCV(model,param_grid)

#train
grid.fit(x_train,y_train)
# print(grid.best_params_)
# print(grid.best_estimator_)

model=grid.best_estimator_

yfit=model.predict(x_test)
# fig,ax=plt.subplots(4,6)
# for i,axi in enumerate(ax.flat):
#     axi.imshow(x_test[i].reshape(62,47),cmap='bone')
#     axi.set(xticks=[],yticks=[])
#     axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
#     color='black' if yfit[i] == y_test[i] else 'blue')
#     # fig.subtitle('Predicted Name')
# plt.show()

# print(classification_report(y_test,yfit,target_names=faces.target_names))

plt.figure(figsize=(9,9))
mat=confusion_matrix(y_test,yfit)
sns.heatmap(mat.T,square=True,
annot=True,fmt='d',cbar=False,
cmap='viridis',
xticklabels=faces.target_names,
yticklabels=faces.target_names
)
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.show()