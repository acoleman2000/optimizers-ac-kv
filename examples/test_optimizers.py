import numpy as np
import pandas as pd
import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import sys, os
sys.path.insert(0, os.path.dirname(os.getcwd()))
sys.path.insert(0,os.getcwd())
os.chdir("..")
from gwu_nn.gwu_network import GWUNetwork
from gwu_nn.layers import Dense
from gwu_nn.activation_layers import Sigmoid


y_col = 'Survived'
x_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
df = pd.read_csv('examples/data/titanic_data.csv')
y = np.array(df[y_col]).reshape(-1, 1)
orig_X = df[x_cols]

# Lets standardize our features
scaler = preprocessing.StandardScaler()
stand_X = scaler.fit_transform(orig_X)
X = stand_X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

network = GWUNetwork()
network.add(Dense(16, add_bias=False, activation='relu', input_size=X.shape[1]))
network.add(Dense(1, add_bias=False, activation='sigmoid'))
network.compile(loss='log_loss', lr=.01)
start = time.time()
network.fit(X_train, y_train, optimizer="GD",batch_size=10, epochs=100)
end = time.time()
print("Time for GD in seconds %ds"%(end - start))
print(network.evaluate(X_train, y_train))
'''
start = time.time()
network.fit(X_train, y_train, optimizer="SGD",batch_size=10, epochs=100)
end = time.time()
print("Time for SGD in seconds %ds"%(end - start))
print(network.evaluate(X_train, y_train))
'''
start = time.time()
network.fit(X_train, y_train, optimizer="ADAM",batch_size=10, epochs=100)
end = time.time()
print("Time for ADAM in seconds %ds"%(end - start))
print(network.evaluate(X_train, y_train))