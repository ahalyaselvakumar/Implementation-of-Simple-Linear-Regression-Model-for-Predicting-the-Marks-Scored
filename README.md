# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1.start

step 2.Import the standard Libraries.

step 3.Set variables for assigning dataset values.

step 4.Import linear regression from sklearn.

step 5.Assign the points for representing in the graph.

step 6.Predict the regression for marks by using the representation of the graph.

step 7.Compare the graphs and hence we obtained the linear regression for the given datas.

step 8.stop
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: AHALYA S
RegisterNumber: 212223230006
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv(r"C:\Users\admin\Downloads\student_scores.csv")
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:
DataSet:
![image](https://github.com/user-attachments/assets/462a10f9-c7e4-406d-be4f-e237e7fc3628)

 Hard Values:
 ![image](https://github.com/user-attachments/assets/9d243411-bb73-4558-8609-ba81d446d6e1)

Tail Values:
![image](https://github.com/user-attachments/assets/b19cb95e-7139-4787-9ddc-ad8a601eebe5)

X and Y Values:
![image](https://github.com/user-attachments/assets/2eeb3591-c044-445d-9e90-7d196a08acd1)

Prediction of X and Y:
![image](https://github.com/user-attachments/assets/b9d1766a-32de-496c-bcf5-739ba6e48f24)

MSE, MAE and RMSE:
![image](https://github.com/user-attachments/assets/c8102715-c10a-4d73-8cd4-7574b707cc7d)

Training Set:
![image](https://github.com/user-attachments/assets/6d0f930d-dfd6-44e0-85be-57c73d616bd5)
![image](https://github.com/user-attachments/assets/2bcce386-ca3e-43a9-bf03-3d2f83dc9176)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
