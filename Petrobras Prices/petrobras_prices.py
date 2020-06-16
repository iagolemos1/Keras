
"""
Created on Mon Jun 15 13:26:54 2020

@author: iago
"""
import pandas as pd
base = pd.read_csv('petro.csv')
base = base.dropna()

base = base.iloc[:,1].values
import matplotlib.pyplot as plt
#%matplotlib inline
plt.plot(base)


#Importing main libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


periods = 30
future_pred = 1

X = base[0:(len(base) - (len(base) % periods))]
X_train = X.reshape(-1, periods, 1)

y = base[1:(len(base) - (len(base) % periods)) + future_pred]
Y_train = y.reshape(-1, periods, 1)

X_test = base[-(periods + future_pred):]
X_test = X_test[:periods]
X_test = X_test.reshape(-1, periods, 1)
Y_test = base[-(periods):]
Y_test = Y_test.reshape(-1, periods, 1)

from model import build_model

model = build_model(future_pred, 200, (X_train.shape[1], 1),
                    'adam', 'mean_squared_error')

model.summary()

model.fit(X_train,Y_train,epochs=500)

predict = model.predict(X_test)

y_teste2 = np.ravel(Y_test)
previsoes2 = np.ravel(predict)


plt.plot(y_teste2, label = 'Valor real')
plt.plot(previsoes2, label = 'Previsões')
plt.legend()
