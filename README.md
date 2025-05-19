# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values
## Program:
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by:ARAVIND KUMAR SS 
RegisterNumber:212223110004
```
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1) #removes the specified row or column.
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x

y=data1['status']
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size =0.2,random_sta

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear')# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)


lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
PLACEMENT DATA:

![image](https://github.com/user-attachments/assets/8508aaf0-f9ef-4424-8128-c082c299caf9)


SALARY DATA:

![image](https://github.com/user-attachments/assets/74e6181f-f839-4b36-9669-3d1721559831)

CHECKING THE NULL() FUNCTION:]

![image](https://github.com/user-attachments/assets/6212b083-1a2b-4210-866a-bc64466314b5)

DATA DUPLICATE:

![image](https://github.com/user-attachments/assets/81f79964-d117-46a2-aa5d-9cf5c9b39322)


PRINT DATA:

![image](https://github.com/user-attachments/assets/2d2cccb0-e494-48fb-85a8-2a59bfceefb7)

DATA_STATUS:

![image](https://github.com/user-attachments/assets/7dbe0249-3180-4570-8cec-287b988f33ac)

DATA_STATUS:

![image](https://github.com/user-attachments/assets/bacd6a46-f1a5-42ff-9c83-98086e82423d)


Y_PREDICTION ARRAY:

![image](https://github.com/user-attachments/assets/1ff11843-5917-431d-997a-409080cbb1d0)


ACCURACY VALUE:

![image](https://github.com/user-attachments/assets/c4911cb6-0e84-4427-abec-a41bd8aa3902)

CONFUSION ARRAY:

![image](https://github.com/user-attachments/assets/f1dd4b79-9002-4f35-9ab0-ef8995941109)


CLASSIFICATION REPORT:

![image](https://github.com/user-attachments/assets/dde0a8ee-c44d-4cc9-ae70-0ecca69b14b2)

PREDICTION OF LR:

![image](https://github.com/user-attachments/assets/fbc4e84d-e3ba-4d46-922a-fb47ced4e8ac)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
