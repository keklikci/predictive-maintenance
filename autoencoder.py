# -*- coding: utf-8 -*-
"""
    File name: autoencoder.py
    Author: kaanguney
    Date created: 10/27/2020
    Date last modified: 10/27/2020
    Python Version: 3.6.9
"""



""" generate the same set of numbers """
from numpy.random import seed
seed(42)
from tensorflow.random import set_seed
set_seed(42)

""" import all libraries """
import sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns


import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Layer
from keras.callbacks import ModelCheckpoint
from keras import regularizers, activations, initializers, constraints, Sequential

from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

""" read data """
import pandas as pd 
df = pd.read_csv("/content/drive/My Drive/Pusula/features4_5m_all.csv")
df.head(1) # first record

print(f"Size of the dataset: {df.shape[0]}\n# of Features: {df.shape[1]}")

""" split train, test, validation data """
train_size = 0.8
train_file = df.iloc[0:int(df.shape[0]*train_size),:]
test = df.iloc[int(df.shape[0]*train_size):,:]
train = train_file.iloc[0:int(train_file.shape[0]*train_size),:]
X_train = train.loc[:,:'ALM_Mainau2_HDVacValveFeed_Scale_PDP_y.1'].values
y_train = train['Label'].values
X_test = test.loc[:,:'ALM_Mainau2_HDVacValveFeed_Scale_PDP_y.1'].values
y_test = test['Label'].values

""" encode and scale data """
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X_train[:,0] = labelencoder.fit_transform(X_train[:,0])
X_test[:,0] = labelencoder.fit_transform(X_test[:,0])

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

""" parameters """
epochs = 100
batch_size = 16
input_dim = X_train_scaled.shape[1] # num of Features 
encoding_dim = 2
learning_rate = 1e-3 # 0.001

""" model """
encoder = Dense(encoding_dim, activation="relu", input_shape=(input_dim,), use_bias = True) 
decoder = Dense(input_dim, activation="sigmoid", use_bias = True)

autoencoder = Sequential()
autoencoder.add(encoder)
autoencoder.add(decoder)

autoencoder.compile(loss='mean_squared_error', optimizer='sgd')
autoencoder.summary()
autoencoder.fit(X_train_scaled, X_train_scaled,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                verbose=0)

train_predictions = autoencoder.predict(X_train_scaled)
print('Train reconstrunction error\n', sklearn.metrics.mean_squared_error(X_train_scaled, train_predictions))
test_predictions = autoencoder.predict(X_test_scaled)
print('Test reconstrunction error\n', sklearn.metrics.mean_squared_error(X_test_scaled, test_predictions))

predictions = autoencoder.predict(X_train_scaled)
mse = np.mean(np.power(X_train_scaled - predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse, 'True_class': y_train})

false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
roc_auc = auc(false_pos_rate, true_pos_rate)

plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
plt.plot([0,1],[0,1], linewidth=5)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

pred_y = [1 if e > 0.010 else 0 for e in error_df.Reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.True_class, pred_y)
LABELS = ["Normal","Abnormal"]
print(classification_report(error_df.True_class, pred_y, target_names=LABELS))
print()
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")