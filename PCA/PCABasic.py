from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
x,y=make_blobs(n_samples=100,n_features=10)

pca=PCA(n_components=4)
pca.fit_transform(x)

df=pd.DataFrame({'var':pca.explained_variance_ratio_,'pc':['PC1','PC2','PC3','PC4']})
sb.barplot(x='pc',y='var',data=df,color='c')
plt.show()

