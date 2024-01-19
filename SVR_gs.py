import numpy as np
import pandas as pd
from statistics import mean, stdev
from sklearn.svm import SVR
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


#Ask for file name and read the file
#file_name = raw_input("Name of file:")
file_name = 'Harvesting data'
data = pd.read_excel(file_name + '.xlsx', header=0)

#Print number of rows and colums read
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")

#Defining X and Y
X = data.drop(columns = ['Water_volume'], axis = 1)
Y = data.Water_volume

#Using Built in train test split function in sklearn
bins = np.linspace(Y.min(), Y.max() + 0.1, 5)
y_binned = np.digitize(Y, bins)


params = {'kernel' : ('sigmoid', 'rbf'),
          'epsilon' : [0.01, 0.05, 0.1, 0.2, 0.5, 1],
          'C' : [0.05, 0.1, 0.5, 1]}

svr = SVR()
    
gs = GridSearchCV(estimator = svr, param_grid = params, scoring = 'r2', cv = 5)


for i in range(50):
    data_train, data_test = train_test_split(data, test_size = 0.2,
                                                stratify = y_binned, random_state = i)
    
    #Hacking a scaling but keeping columns names since min_max_scaler does not return a dataframe
    minval = data_train.min()
    minmax = data_train.max() - data_train.min()
    data_train_scaled = (data_train - minval) / minmax
    data_test_scaled = (data_test - minval) / minmax
    
    #Define X and Y
    X_train = data_train_scaled.drop(columns = ['Water_volume'], axis=1)
    Y_train = data_train_scaled.Water_volume
    X_test = data_test_scaled.drop(columns = ['Water_volume'], axis=1)
    Y_test = data_test_scaled.Water_volume
    
    
    # fitting the model for grid search
    grid_result = gs.fit(X_train, Y_train)
    
    print('Best score = {:.4f} using {}'.format(gs.best_score_,
                                            gs.best_params_))

    
