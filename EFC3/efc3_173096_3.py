#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

np.set_printoptions(precision = 4)


# In[2]:


def Formata_Matriz(mc):
    
    Colunas = {"Basófilos": mc[:, 0],
    "Eosinófilos": mc[:, 1] ,
    "Eritroblastos": mc[:, 2],
    "Granulócitos Imaturos": mc[:, 3],
    "Linfócitos": mc[:, 4],
    "Monócitos": mc[:, 5],
    "Neutrófilos": mc[:, 6],
    "Plaquetas": mc[:, 7]}
    Indice = ["Basófilos", "Eosinófilos", "Eritroblastos", "Granulócitos Imaturos", "Linfócitos", "Monócitos", "Neutrófilos", "Plaquetas"]
    mc_df = pd.DataFrame(Colunas, index = Indice)
    
    return mc_df

def One_Hot_Encoding(Labels):
    One_Hot_Labels = []
    for Label in Labels:
        One_Hot_Label = np.zeros(8)
        One_Hot_Label[Label] = 1
        One_Hot_Labels.append(One_Hot_Label)
    return np.array(One_Hot_Labels)


# In[3]:


Train_Images = np.load('train_images.npy')/255
Train_Labels = np.load('train_labels.npy')

Val_Images = np.load('val_images.npy')/255
Val_Labels = np.load('val_labels.npy')

Test_Images = np.load('test_images.npy')/255
Test_Labels = np.load('test_labels.npy')

One_Hot_Train = One_Hot_Encoding(Train_Labels)
One_Hot_Val = One_Hot_Encoding(Val_Labels)


# Para o item d) foi proposta uma camada convolucional adicional com uma redução na dimensão
# do Kernel, a fim de não reduzir tanto a dimensão dos feature maps resultantes da convolução.
# Além disso, o número de Kernels de cada camada foi aumentado para 15, para que mais feature
# maps fossem gerados. A camada de Pooling permaneceu sendo do tipo MaxPooling, contudo após
# a camada de Flatten foi feito um processo de Dropout. O Dropout consiste em atribuir uma
# probabilidade para que os neurônios sejam desativados a cada passo de treinamento. Isto contribui
# para que cada neurônio desempenhe um papel útil para a rede por conta própria, diminuindo a
# dependencia dos seus vizinhos. No caso desta atividade a probabilidade que resultou em um melhor
# desempenho foi de 0.01. Usando esta configuração a rede neural conseguiu atingir uma acurácia
# de 90% junto aos dados de teste em 20 épocas de treinamento, representando um ganho de 6% em
# relação aos itens c) e a).

# In[6]:


CNN = Sequential()

CNN.add(Conv2D(filters = 15, 
               kernel_size = (2, 2),
               activation ='relu',
               input_shape = (28,28,3)))

CNN.add(Conv2D(filters = 15, 
               kernel_size = (2, 2),
               activation ='relu',
               input_shape = (28,28,3)))


CNN.add(MaxPooling2D(pool_size=(5, 5)))

CNN.add(Flatten())

CNN.add(Dropout(0.01))

CNN.add(Dense(8, activation='softmax'))


CNN.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(learning_rate = 1e-2),
            metrics = ['accuracy'])

CNN.fit(Train_Images, One_Hot_Train,
        batch_size = 100,
        epochs = 20,
        verbose = 0,
        validation_data=(Val_Images, One_Hot_Val))

Preds = []
Soft_Preds = CNN.predict(Test_Images, verbose = 0)
for Pred in Soft_Preds:
    Preds.append(np.argmax(Pred))
    
print("Acurácia:", accuracy_score(Preds, Test_Labels))

CM = confusion_matrix(Test_Labels, Preds, labels=[0, 1, 2, 3, 4, 5, 6, 7])
Formata_Matriz(CM)



# In[5]:


i = 0
for Pred, Test_Label, Soft_Pred in zip(Preds, Test_Labels, Soft_Preds):
    if Pred != Test_Label and i <= 5:
        print("Classe Esperada:", Test_Label[0])
        print("Classe Predita:", Pred)
        print("Probabilidades", np.array(Soft_Pred))
        print("\n\n")
        i += 1

