import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('../Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/Multiple_Linear_Regression/50_Startups.csv')
#print (dataset)
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#categorical Data
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

X=X[:,1:]

#Splitting data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Backward Elimination- to remove the features that are not very important
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print (regressor_OLS.summary())
    return x

SL = 0.07
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt = X[:, [0, 1, 2, 3, 4,5]]
X_Modeled = backwardElimination(X_opt, SL)


#Linear regression calculates an equation that minimizes the distance between the fitted line and all of the data points.
#echnically, ordinary least squares (OLS) regression minimizes the sum of the squared residuals.
#In general, a model fits the data well if the differences between the observed values and the model's
#predicted values are small and unbiased.

#R-squared is a statistical measure of how close the data are to the fitted regression line.
#It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression.

#The definition of R-squared is fairly straight-forward; it is the percentage of the response variable variation that is
#explained by a linear model. Or:

#R-squared = Explained variation / Total variation

#R-squared is always between 0 and 100%:

#    0% indicates that the model explains none of the variability of the response data around its mean.
#    100% indicates that the model explains all the variability of the response data around its mean.

#In general, the higher the R-squared, the better the model fits your data.

#Limitations:
#R-squared cannot determine whether the coefficient estimates and predictions are biased,
#which is why you must assess the residual plots.

#R-squared does not indicate whether a regression model is adequate. You can have a low R-squared
#value for a good model, or a high R-squared value for a model that does not fit the data!


#Using R-squared value
'''
import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue

    regressor_OLS.summary()
    return x

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

'''
