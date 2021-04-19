# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 15:48:21 2021

@author: Soorya Parthiban
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import pickle

##-----------------------------------------------------------------------------
## Importing the dataset
##-----------------------------------------------------------------------------'

fish_df = pd.read_csv(r"D:\001_Data\Regression Android\Fish Market\Fish.csv")

fish_df.head()
fish_df.tail()
fish_df.info()
fish_df.describe()
fish_df.isnull().sum()
fish_df.shape

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

fish_df["Outcome"] = le.fit_transform(fish_df["Species"])
fish_df["Outcome"].value_counts()
fish_df["Outcome"].nunique()

le.classes_
outcome_variables = ['Bream', 'Parkki', 'Perch',
                     'Pike', 'Roach', 'Smelt', 'Whitefish']

fish_df.columns

df = fish_df[['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width',
       'Outcome']]

df.shape

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

##-----------------------------------------------------------------------------
## Dataset Pre-Processing & Partitioning
##-----------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

print(X_train.shape, X_test.shape)

##-----------------------------------------------------------------------------
## Building the DL Algorithms
##-----------------------------------------------------------------------------

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

## MODEL 1

model_1 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(6,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(7)
])

model_1.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])

model_1.summary()

model_1.fit(X_train, y_train, epochs=500)

##-----------------------------------------------------------------------------

## MODEL 2

model_2 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(6,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(7)
])

model_2.compile(loss=loss_fn, optimizer=tf.keras.optimizers.Adamax(learning_rate=0.1), 
                metrics=['accuracy'])

model_2.summary()

model_2.fit(X_train, y_train, epochs=500)

##-----------------------------------------------------------------------------

## MODEL 3

model_3 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(6,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(7)
])

model_3.compile(loss=loss_fn, optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.1), 
                metrics=['accuracy'])

model_3.summary()

model_3.fit(X_train, y_train, epochs=500)


##-----------------------------------------------------------------------------
## Model Evaluation
##-----------------------------------------------------------------------------

from sklearn.metrics import accuracy_score, confusion_matrix

y_preds = model_2.predict_classes(X_test)

print("The Accuracy Score of the Fish Classification Model: " ,accuracy_score(y_test, y_preds))

cm = confusion_matrix(y_test, y_preds)

x_axis_labels = ['Bream', 'Parkki', 'Perch', 'Pike', 'Roach', 'Smelt', 'Whitefish']
y_axis_labels = ['Bream', 'Parkki', 'Perch', 'Pike', 'Roach', 'Smelt', 'Whitefish']

f, ax = plt.subplots(figsize =(15,10))
sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Greens", 
            xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for the Fish Classification Model')


##-----------------------------------------------------------------------------
## Saving the Model
##-----------------------------------------------------------------------------

Keras_file = "Fish-Classification.h5"

tf.keras.models.save_model(model_2, Keras_file)
converter = tf.lite.TFLiteConverter.from_keras_model(model_2)
tfmodel = converter.convert()
open("Fish-Classification-Model.tflite","wb").write(tfmodel)