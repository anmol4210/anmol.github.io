import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading dataset
dataset=pd.read_csv("../Machine Learning A-Z Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression/Polynomial_Regression/Position_Salaries.csv")
X=dataset.iloc[:,1:2].values
Y=dataset.iloc[:,2].values

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
#poly_reg.fit(X_poly,Y)
lin_reg=LinearRegression()
lin_reg.fit(X_poly,Y)
Y_predict=lin_reg.predict(poly_reg.fit_transform(X))



plt.scatter(X,Y,color='red')
plt.plot(X,Y_predict,color='blue')
plt.show()


