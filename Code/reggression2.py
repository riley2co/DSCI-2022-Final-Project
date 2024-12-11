
#libraries 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import sklearn as sk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import psutil
import time


dummyDf = pd.read_csv('/scratch/ptolemy/users/rnt89/FinalProject/monstersNew.csv')
start = time.time()

X = dummyDf.drop(['cr'],axis=1)
Y = dummyDf['cr']

trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2, random_state = 123)


rfModel = RandomForestRegressor(n_estimators=250, random_state=42,max_depth=10)
rfModel.fit(trainX, trainY)

pred = rfModel.predict(testX)

print("Mean Squared Error: ",mean_squared_error(testY, pred))

plt.scatter(testY, pred)

plt.plot([testY.min(), testY.max()], [testY.min(), testY.max()], 'k--', label='Perfect prediction line')

plt.xlabel('Actual Values')

plt.ylabel('Predicted Values')
plt.grid()
plt.title('Random Forest Regression Predictions')
plt.savefig('multiCore.png')


end = time.time()
print(f"Execution time: {end - start} seconds")
print(f"Total CPU Usage: {psutil.cpu_percent()}%")
