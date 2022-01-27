#Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings('ignore') 

#Load dataset
df = pd.read_csv("D:/Loan_prediction_dataset/train_u6lujuX_CVtuZ9i.csv")
test = pd.read_csv("D:/Loan_prediction_dataset/test_Y3wMUE5_7gLdaTN.csv")

#preprocessing data
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].median())
df.dropna(inplace=True)

#Label encoding
df.Loan_Status=df.Loan_Status.map({'Y':1,'N':0})
df.Gender=df.Gender.map({'Male':1,'Female':0})
df.Married=df.Married.map({'Yes':1,'No':0})
df.Dependents=df.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
df.Education=df.Education.map({'Graduate':1,'Not Graduate':0})
df.Self_Employed=df.Self_Employed.map({'Yes':1,'No':0})
df.Property_Area=df.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

df = df.drop(columns = ['Loan_ID'] , axis = 1)

X = df.iloc[:, :11]
y = df.iloc[:, -1]

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

df = df.drop(columns = ['Loan_Status'] , axis = 1)

from sklearn.model_selection import cross_val_score
def classify(model, x, y):
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
    model.fit(x_train,y_train)
    print("Accuracy is",model.score(x_test,y_test)*100)
    score = cross_val_score(model, x, y, cv=5)
    print("Cross validation is",np.mean(score)*100)

model = LogisticRegression()
classify(model,X,y)

# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))