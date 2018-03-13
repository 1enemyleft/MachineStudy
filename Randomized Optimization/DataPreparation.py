#!/Users/jiananduan/anaconda/bin/python

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy 

datatrain = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
# assign names to columns
datatrain.columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
       'petal width (cm)', 'species']

# Change string value to numeric
datatrain.set_value(datatrain['species']=='Iris-setosa',['species'],0)
datatrain.set_value(datatrain['species']=='Iris-versicolor',['species'],1)
datatrain.set_value(datatrain['species']=='Iris-virginica',['species'],2)
datatrain = datatrain.apply(pd.to_numeric)
datatrain_array = datatrain.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(datatrain_array[:,:4],
                                                    datatrain_array[:,4],
                                                    test_size=0.3)

numpy.savetxt("X_train.csv", X_train, delimiter=",")
numpy.savetxt("X_test.csv", X_test, delimiter=",")
numpy.savetxt("y_train.csv", y_train, delimiter=",")
numpy.savetxt("y_test.csv", y_test, delimiter=",")
