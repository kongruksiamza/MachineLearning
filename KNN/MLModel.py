import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
rng=np.random

#การจำลองข้อมูล
x=rng.rand(50)*10
y=2*x+rng.randn(50)

#linear regression model
model=LinearRegression()
x_new =x.reshape(-1,1)

#train
model.fit(x_new,y)

#test model
xfit=np.linspace(-1,11)
xfit_new=xfit.reshape(-1,1)

yfit=model.predict(xfit_new)

#analysis model & result
plt.scatter(x,y)
# plt.plot(xfit,yfit)
plt.show()

