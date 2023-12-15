
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import svm 
from sklearn import metrics
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import os
def main():


    data = pd.read_csv('data/iris/Iris.csv')
    data = data.drop('Id', axis=1, errors='ignore')
    encoder = LabelEncoder()
    data['Species'] = encoder.fit_transform(data['Species'])
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    os.makedirs('data/prepared',exist_ok=True)
    homedir = os.path.expanduser("./")
    np.save(os.path.join(homedir,'data/prepared/X_train.npy'),X_train)
    np.save(os.path.join(homedir,'data/prepared/X_test.npy'),X_test)
    np.save(os.path.join(homedir,'data/prepared/y_train.npy'),y_train)
    np.save(os.path.join(homedir,'data/prepared/y_test.npy'),y_test)
if __name__ == "__main__":
    main()
