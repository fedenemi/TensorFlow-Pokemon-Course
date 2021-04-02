import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

def dummy_creation(df, dummy_categories):
    for i in dummy_categories:
        df_dummy = pd.get_dummies(df[i])
        df = pd.concat([df,df_dummy],axis=1)
        df = df.drop(i, axis=1)
    return(df)

def train_test_splitter(DataFrame, column):
    df_train = DataFrame.loc[df[column] != 1]
    df_test = DataFrame.loc[df[column] == 1]

    df_train = df_train.drop(column, axis=1)
    df_test = df_test.drop(column, axis=1)

    return(df_train, df_test)

def label_delineator(df_train, df_test, label):
    
    train_data = df_train.drop(label, axis=1).values
    train_labels = df_train[label].values
    test_data = df_test.drop(label,axis=1).values
    test_labels = df_test[label].values
    return(train_data, train_labels, test_data, test_labels)

def data_normalizer(train_data, test_data):

    train_data = preprocessing.MinMaxScaler().fit_transform(train_data)

    test_data = preprocessing.MinMaxScaler().fit_transform(test_data)

    return(train_data, test_data)    

#upload csv and put the part that we want in the dataframe

df = pd.read_csv(r'C:\Users\Fede\Documents\GitHub\TensorFlow-Pokemon-Course\Project_files\pokemonez.csv')

df = df[['isLegendary','Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed','Color','Egg_Group_1','Height_m','Weight_kg','Body_Style']]

#transform it to a useful data type

df = dummy_creation(df, ['isLegendary'])
df = dummy_creation(df, ['Egg_Group_1', 'Body_Style', 'Color','Type_1', 'Type_2'])

#split the data

df_train, df_test = train_test_splitter(df, 'Generation')

train_data, train_labels, test_data, test_labels = label_delineator(df_train, df_test, 'isLegendary')    





#Normalize and model data

train_data, test_data = data_normalizer(train_data, test_data)  

length = train_data.shape[1]
model = keras.Sequential()

model.add(keras.layers.Dense(500, activation='relu', input_shape=[length]))

model.add(keras.layers.Dense(2, activation='softmax')) 