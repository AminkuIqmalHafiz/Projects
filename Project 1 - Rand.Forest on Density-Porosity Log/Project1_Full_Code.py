import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error

#Extracting dataframe
raw = pd.read_csv("C:/Users/Aminku Iqmal/Downloads/log.csv")
#checking the collinearity through correlation heatmap
features = ["Depth", "RxoRt", "RLL3", "SP", "RILD", "MN", "MI", "MCAL", "DCAL", "CNLS", "GR"]
sns.heatmap(data=raw[features].corr(),annot=True)
plt.title("Pearson Correlation Heatmap")
plt.show()

#define variables
filtered_features = ["Depth","RxoRt", "RLL3", "SP", "RILD", "MN","DCAL", "CNLS", "GR"]
X = raw[filtered_features]
Y = raw["DPOR"]
print(X.head())
print(Y.head())

#data splitting
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.transform(X_test)

#model selection and training, gridsearchcv for hyperparameter tuning
rf = RandomForestRegressor(random_state=0)
param_grid = {
    'n_estimators' : [10,20,100,200,300],
    'max_depth' : [None,10,20],
    'min_samples_split' : [2,5,10],
    'min_samples_leaf' : [2,3,4],
    'max_features' : ['log2','sqrt',None]
}

grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)
grid_search.fit(x_train,y_train)

#training results
print("Best Parameters:", grid_search.best_params_)
print("Train R2 Score:", grid_search.best_estimator_.score(x_train,y_train))
print("MSE(Train):", mean_squared_error(y_train, grid_search.best_estimator_.predict(x_train)))

#testing results
y_pred = grid_search.best_estimator_.predict(x_test)

print("---------------------------------")
print("Test R2 Score:", r2_score(y_test, y_pred))
print("MSE(Test):", mean_squared_error(y_test, y_pred))

"""
Best Parameters: {'max_depth': None, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}
Train R2 Score: 0.9947914850535767
MSE(Train): 3.0780074613006456
---------------------------------
Test R2 Score: 0.9742396005291164
MSE(Test): 15.220790477057761
"""





