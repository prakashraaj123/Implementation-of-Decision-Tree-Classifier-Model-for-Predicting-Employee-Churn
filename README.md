# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Start the program
2. import pandas and the required "Employee.csv" file
3. print data.head(),data.info() and count the values in "left" column
4. import LabelEncoder and transform values in "salary" column
5. assign x values,y values from dataset and import the train_test_split
6. split the dataset into train and test data where test_size=0.2 and random_state=100
7. after splitting dataset , import DecisionTreeClassifier and import metrics
8. finally use dt.predict()
9. Stop the program.
 
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:S.prakash Raaj 
RegisterNumber:212220040120  
*/
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]s])

```

## Output:
![decision tree classifier model](/datahead.PNG)
![decision tree classifier model](/info.PNG)
![decision tree classifier model](/isnullsum.PNG)
![decision tree classifier model](/dataheads.PNG)
![decision tree classifier model](/xheadnw.PNG)
![decision tree classifier model](/accuracy.PNG)
![decision tree classifier model](/predict.PNG)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
