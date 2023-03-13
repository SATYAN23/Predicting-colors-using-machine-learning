
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv("data.csv")
encoder = LabelEncoder()
encoder.fit(dataset['label'])
Y = encoder.transform(dataset['label'])
X=dataset.iloc[:, 0:3].values

trainX,testX,trainY,testY=train_test_split(X,Y,test_size=0.3,random_state = 42)

trainX=np.array(trainX)
trainY=np.array(trainY).astype('int64')
testX=np.array(testX)


model = linear_model.LogisticRegression(class_weight="balanced")
model.fit(trainX,trainY)

z=model.predict([[127,67,254]])
z=z.astype("int64")
y1=accuracy_score([5],z)
print("accuracy score"+" : "+ str(y1))

if 0.9<z<=1:
  z=1
elif z==0:
  z=0
elif 0.8<z<0.9:
  z=2
elif 0.7<z<0.8:
  z=3
elif 0.6<z<0.7:
  z=4
elif 0.5<z<0.6:
  z=5
elif 0.4<z<0.5:
  z=6
elif 0.3<z<0.4:
  z=7
elif 0.2<z<0.3:
  z=8
elif 0.1<z<0.2:
  z=9
elif 0<z<0.1:
  z=10

  
z=encoder.inverse_transform([z])
print(z)
