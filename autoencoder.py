# -*- coding: utf-8 -*-
"""
    File name: autoencoder.py
    Author: kaanguney
    Date created: 10/27/2020
    Date last modified: 10/29/2020
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
df.head(1) # fetch first

print(f"Size of the dataset: {df.shape[0]}\n# of Features: {df.shape[1]}")

""" split train, test, validation data """

df.drop(['TimeStamp_x'], axis = 1, inplace = True)

train_size = 0.8
train_valid_size = 0.8
train_file = df.iloc[0:int(df.shape[0]*train_size),:]
test = df.iloc[int(df.shape[0]*train_size):,:]
train = train_file.iloc[0:int(train_file.shape[0]*train_size),:]
valid = train_file.iloc[int(train_file.shape[0]*train_valid_size):,:]

X_train = train.loc[:,:'ALM_Mainau2_HDVacValveFeed_Scale_PDP_y.1'].values
y_train = train['Label'].values
X_test = test.loc[:,:'ALM_Mainau2_HDVacValveFeed_Scale_PDP_y.1'].values
y_test = test['Label'].values
X_valid = valid.loc[:,:'ALM_Mainau2_HDVacValveFeed_Scale_PDP_y.1'].values
y_valid = valid['Label'].values

""" before rearranging data """
print(f"Train features shape: {X_train.shape}\t Train label shape: {y_train.shape}")
print(f"Test features shape: {X_test.shape}\t Train label shape: {y_test.shape}")
print(f"Validation features shape: {X_valid.shape}\t Validation label shape: {y_valid.shape}")

""" encode and scale """
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X_train[:,0] = labelencoder.fit_transform(X_train[:,0])
X_test[:,0] = labelencoder.fit_transform(X_test[:,0])
X_valid[:,0] = labelencoder.fit_transform(X_valid[:,0])

""" prepare data """
X_train_y0 = X_train[y_train == 0]
X_valid_y0 = X_valid[y_valid == 0]

scaler = MinMaxScaler()
scaler.fit(X_train_y0)
X_train_y0_scaled = scaler.transform(X_train_y0)
X_valid_y0_scaled = scaler.transform(X_valid_y0)

""" scale validation and test data """
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

""" parameters """
epochs = 100
batch_size = 16
input_dim = X_train_y0_scaled.shape[1] # num of Features 
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
autoencoder_history = autoencoder.fit(X_train_y0_scaled, X_train_y0_scaled,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_valid_y0_scaled, X_valid_y0_scaled),
                shuffle=True,
                verbose=0)

plt.plot(autoencoder_history.history['loss'], linewidth=2, label='Train')
plt.plot(autoencoder_history.history['val_loss'], linewidth=2, label='Valid')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

""" validation predictions for threshold selection """
x_valid_pred = autoencoder.predict(X_valid_scaled)
mse = np.mean(np.power(X_valid_scaled - x_valid_pred, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse, 'True_class': y_valid})

print('Validation reconstrunction error\n', sklearn.metrics.mean_squared_error(X_valid_scaled, x_valid_pred))

""" testing reconstruction error """
x_test_pred = autoencoder.predict(X_test_scaled)
print('Test reconstrunction error\n', sklearn.metrics.mean_squared_error(X_test_scaled, x_test_pred))

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

""" Plot reconstruction errors for positive and negative data points """
mse = np.mean(np.power(X_test_scaled - x_test_pred, 2), axis=1)

error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': y_test})
threshold = round(np.mean(error_df.Reconstruction_error),2)
print(f"Threshold selected is: {threshold}")

""" Reconstruction error for different classes """
groups = error_df.groupby('True_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Abnormal" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show()

""" Plot ROC Curve """

LABELS = ["Normal","Abnormal"]

pred_y = [1 if e > threshold else 0 for e in error_df.Reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.True_class, pred_y)

plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

""" plot the ROC curve again """
false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df.True_class, error_df.Reconstruction_error)
roc_auc = auc(false_pos_rate, true_pos_rate,)

plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
plt.plot([0,1],[0,1], linewidth=5)

plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
