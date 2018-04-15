import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading Data
dataset=pd.read_csv('../Machine Learning A-Z Template Folder/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv')
#  Y- Dependent and  X-Independent data
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values
#print (X)
#print (Y)

#Splitting Data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

#print (X_test)
from sklearn.linear_model import LinearRegression

#Training the model
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#predicting the values
y_pred=regressor.predict(X_test)

print(y_pred)

#Saving result
csv = open("Output.csv", "w")
csv.write(("Actual,Predicted\n"))
for i in range(len(y_pred)):
    csv.write(str(Y_test[i])+","+str(y_pred[i])+'\n')

plt.scatter(X_train,Y_train,color='red')

plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Test Data
plt.scatter(X_test,Y_test,color='red')
plt.scatter(X_test,y_pred,color='yellow')

plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Testing set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
