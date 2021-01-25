import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score

def load_data():
    #load the data 
    df = pd.read_csv("try.csv",names=['Temperature','Precipitation','Humidity','Wind'], header=None)
    #drop the data if the data is empty
    df=df.dropna()
    #selete the feature we want
    features=['Precipitation','Humidity','Wind']
    X=df[features].values
    y=df['Temperature'].values
    return X, y,features

def spilt(X,y):
    #spilt X y into train and test
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 5)
    return X_train, X_test, y_train, y_test

def show(y_pred,y_test):
    #show the predict and the real data
    plt.plot(y_pred[:150], label = 'prediction')
    plt.plot(y_test[:150], label = 'data')
    plt.legend()
    plt.show()

def seletebestdepth(X,y):
    #run depth from 1 to 9 to get the best depth
    all_scores = []
    best_score = -1
    best_depth = 0
    for i in range(1,9):
        treereg = DecisionTreeRegressor(max_depth=i, random_state=1)
        scores = -cross_val_score(treereg, X, y, scoring='neg_mean_absolute_error', cv=10)
        current_score = np.mean(np.sqrt(scores.mean()))
        if current_score < best_score or best_score == -1:
            best_score = current_score
            best_depth = i
        all_scores.append(current_score)
        print("max depth: ",i," MSE: ",current_score)
    #print("best score :", best_score)
    #print("best depth :", best_depth)
    #plt.figure()
    #plt.plot(range(1, 9), all_scores)
    #plt.xlabel('x=max tree depth')
    #plt.show()
    return best_depth

def test(reg,X_train, y_train):
    #choose the best parmeter
    parameters_dtr = {"criterion" : ["mse", "friedman_mse", "mae"],   
                    "splitter" : ["best", "random"], 
                    "min_samples_split" : [2, 3, 4, 5, 6, 7, 8, 9, 10],  
                    "max_features" : ["auto", "log2"]}
    grid_dtr = GridSearchCV(reg, parameters_dtr, verbose=1, scoring="r2")
    #grid_dtr.fit(X_train, y_train)
    #print("Best DecisionTreeRegressor Model: " + str(grid_dtr.best_estimator_))
    return grid_dtr

if __name__ == "__main__":
    #load the data 
    X,y,features=load_data();
    y=y
    X_train, X_test, y_train, y_test=spilt(X,y)

    #selete best depth
    best_depth = seletebestdepth(X,y)
    print(best_depth)
    #load the model
    reg1 = DecisionTreeRegressor(max_depth=best_depth)
    reg1.fit(X_train, y_train)
    y_pred = reg1.predict(X_test)

    #let the best depth to run test fun 
    #tune the parameter
    reg2 = test(reg1,X_train,y_train)
    reg2.fit(X_train, y_train)

    #best regressor
    bestreg = DecisionTreeRegressor()
    bestreg=reg2
    y_pred=bestreg.predict(X_test)

    #show(y_pred,y_test)
    print('RMSE of Decision Tree Regression:',np.sqrt(mean_squared_error(y_pred,y_test)))
    #rmse = np.sqrt(mean_squared_error(y_pred, y_test))
    #print('Test RMSE: %.3f' % rmse)

    while True:
        print ("輸入feature: ")
        input_x=input("Precipitation: ")
        if input_x=="exit":
            break
        input_y=input("Humidity: ")
        input_z=input("Wind: ")
        X_today = [[input_x,input_y,input_z]]
        y_today = bestreg.predict(X_today)
        print("temperature is ",y_today)