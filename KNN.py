import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

path = 'breast-cancer-wisconsin.data'
df = pd.read_csv(path)
df.replace('?', -99999, inplace = True) #separating the outliers
df.drop(['id'], 1, inplace = True) # Dumping the id column as it is useless

X = np.array(df.drop(['class'], 1))
Y = np.array(df['class'])

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size = 0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print(accuracy)
