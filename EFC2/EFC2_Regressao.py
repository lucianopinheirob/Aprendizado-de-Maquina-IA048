#!/usr/bin/env python
# coding: utf-8

# In[212]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision = 4)

def Conta_Classes(Label_Set):
    P = np.sum(np.equal(Label_Set, 1))
    N = np.sum(np.equal(Label_Set, 0))
    return np.array([P/(P+N), N/(P+N)])

Train_Images = np.load('train_images.npy')
Train_Labels = np.load('train_labels.npy')

Val_Images = np.load('val_images.npy')
Val_Labels = np.load('val_labels.npy')

Test_Images = np.load('test_images.npy')
Test_Labels = np.load('test_labels.npy')


#Conta_Classes(Train_Labels)
#Conta_Classes(Val_Labels)
#Conta_Classes(Test_Labels)


# O balanço entre classes, obtido pela função *Conta_Classes* foi igual para os três conjuntos - treino, validação e teste:
# 
# - Positivo: **73.08%**
# - Negativo: **26.92%**
# 
# Há uma predominância da classe positiva, aproximadamente 3 em cada 4 amostras pertencem a esta classe.

# In[2]:


#Extração de atributos a partir das imagens

def Atributos(Image_Set, Dim_Image):
    Atributos_Test_Empilhados = np.empty((0, 88))
    for image in Image_Set:
        Soma = np.sum(image) / (Dim_Image ** 2)
        Traco = np.trace(image)
        Autovalores = np.linalg.eig(image)[0]
        Autovalores = np.absolute(Autovalores)
        Soma_Lin = np.array([0])
        Soma_Col = np.array([0])
        for lin in image:
            Soma_Lin = np.hstack((Soma_Lin, np.sum(lin) / Dim_Image))
        for col in image.T:
            Soma_Col = np.hstack((Soma_Col, np.sum(col) / Dim_Image))
        Atributos_Test = np.hstack((Soma, Autovalores, Traco, Soma_Lin, Soma_Col))
        Atributos_Test_Empilhados = np.vstack((Atributos_Test_Empilhados, Atributos_Test)) 
    Fi = np.hstack((np.ones((len(Atributos_Test_Empilhados), 1)), Atributos_Test_Empilhados))
    return Fi
    

Dim_Image = len(Train_Images[0])
Fi_Train = Atributos(Train_Images, Dim_Image)
Fi_Val   = Atributos(Val_Images, Dim_Image)
Fi_Test  = Atributos(Test_Images, Dim_Image)


# A função Atributos, foi utilizada para extrair os seguintes atributos para cada imagem:
# 
# - Média da soma dos elementos.
# - Traço da matriz.
# - Valores absolutos dos autovalores.
# - Média da soma dos elementos de cada linha.
# - Média da soma dos elementos de cada coluna.
# 
# Outros atributos, como determinante e parte real dos autovalores, foram testados, contudo, os atributos acima foram os que proporcionaram os melhores resultados. Os atributos de cada imagem foram empilhados e a matriz Fi de cada conjunto - treino, validação e teste - foi gerada.

# In[45]:


#Treinamento
def Calcula_Grad(Fi_Train, w, Train_Labels):
    E = np.empty((0, 1))
    for i in range(len(Fi_Train)):
        Label_Calc = 1 / (1 + np.exp(-(np.dot(Fi_Train[i], w))))
        Erro = Train_Labels[i][0] - Label_Calc
        E = np.vstack((E, Erro))
    Grad = - 1/len(Fi_Train) * (np.matmul(E.T, Fi_Train))
    return Grad[0]

w = np.loadtxt('Melhor_Parametro') 

#w = [1]
#w = len(Fi_Train[0])*w
#w = np.array(w)

#w = np.arange(0, 1, len(Fi_Train[0]))

Alfa = .0001
for i in range(3000):
    w  = w - Alfa * Calcula_Grad(Fi_Train, w, Train_Labels)
    if i % 1000 == 0: 
        print(i)
        


# Para o treinamento foi utilizado o método do gradiente descendente, em batelada, utilizando a função *Calcula_Grad*. A escolha é coerente com base no tamanho do conjunto de dados deste problema. O passo $\alpha$ e o número de épocas de treinamento foram ajustados com base nos melhores valores de Acurácia Balanceada e Precisão alcançados na etapa de validação. Isto foi obtido com $\alpha = .0001$ e $10000$ épocas. Além disso, foi observado que o valor inicial do vetor de parâmetros também altera fundamentalmente os resultados. Os melhores resultados foram obtios com uma inicialização aleatória, seguindo uma distribuição uniforme no conjunto $[0, 1)$.

# In[46]:


#Validacao
Label_Calcs = np.empty((0,1))
for i in range(len(Fi_Val)):
    Label_Calc = 1 / (1 + np.exp(-(np.dot(Fi_Val[i], w))))
    if Label_Calc >= 0.5:
        Label_Calc = 1
    else:
        Label_Calc = 0
    Label_Calcs = np.vstack((Label_Calcs, Label_Calc))
    

TP = np.sum(np.equal(Label_Calcs, 1) & np.equal(Val_Labels, 1))
FP = np.sum(np.equal(Label_Calcs, 1) & np.equal(Val_Labels, 0))
FN = np.sum(np.equal(Label_Calcs, 0) & np.equal(Val_Labels, 1))
TN = np.sum(np.equal(Label_Calcs, 0) & np.equal(Val_Labels, 0))
    

Matriz_Conf = np.array([[TP, FN], 
                        [FP, TN]])    
    
Espe = TN / (TN + FP)
Sens = TP / (TP + FN)
Prec = TP / (TP + FP)
TFPo = FP / (TN + FP)

BA = (Espe + Sens) / 2
Acc = (TP + TN) / (TP + TN + FP + FN)
Fm = (2 * Sens * Prec) / (Sens + Prec)

BA


# Para a validação foi utilizado o vetor de parâmetros obtido na etapa de treinamento. O limite utilizado para a escolha da classe foi de 0.5. Se o padrão calculado for maior que 0.5 o classificador escolhe classe positiva, caso contrário, escolhe classe negativa. A matriz de confusão foi calculada e com base nela as métricas foram calculadas.

# In[213]:


#Teste
Label_Calcs = np.empty((0,1))
Label_Calcs_Bruto = np.empty((0,1)) #Será usado para curva ROC
for i in range(len(Fi_Test)):
    Label_Calc = 1 / (1 + np.exp(-(np.dot(Fi_Test[i], w))))
    Label_Calcs_Bruto = np.vstack((Label_Calcs_Bruto, Label_Calc))
    if Label_Calc >= 0.5:
        Label_Calc = 1
    else:
        Label_Calc = 0
    Label_Calcs = np.vstack((Label_Calcs, Label_Calc))
    


TP = np.sum(np.equal(Label_Calcs, 1) & np.equal(Test_Labels, 1))
FP = np.sum(np.equal(Label_Calcs, 1) & np.equal(Test_Labels, 0))
FN = np.sum(np.equal(Label_Calcs, 0) & np.equal(Test_Labels, 1))
TN = np.sum(np.equal(Label_Calcs, 0) & np.equal(Test_Labels, 0))
    

Matriz_Conf = np.array([[TP, FN], 
                        [FP, TN]])    
    
Espe = TN / (TN + FP)
Sens = TP / (TP + FN)
Prec = TP / (TP + FP)
TFPo = FP / (TN + FP)

BA = (Espe + Sens) / 2
Acc = (TP + TN) / (TP + TN + FP + FN)
Fm = (2 * Sens * Prec) / (Sens + Prec)

BA


# In[196]:


#Curva ROC

Sens = []
TFPo = []
Thresholds = np.logspace(-100, 0, num=100-0+1,base=10,dtype='float128')

#Thresholds = np.array([1])

for Threshold in Thresholds:
    
    Label_Calc_ROC = np.empty((0,1))
    for Label_Calc_Bruto in Label_Calcs_Bruto:
        #print(Label_Calc_Bruto)
        if Label_Calc_Bruto > Threshold:
            Label_Calc_Bruto = 1
        else:
            Label_Calc_Bruto = 0
        Label_Calc_ROC = np.vstack((Label_Calc_ROC, Label_Calc_Bruto))
    #print(Threshold)
    #print(np.hstack((Label_Calcs_Bruto, Label_Calc_ROC)))

    TP = np.sum(np.equal(Label_Calc_ROC, 1) & np.equal(Test_Labels, 1))
    FP = np.sum(np.equal(Label_Calc_ROC, 1) & np.equal(Test_Labels, 0))
    FN = np.sum(np.equal(Label_Calc_ROC, 0) & np.equal(Test_Labels, 1))
    TN = np.sum(np.equal(Label_Calc_ROC, 0) & np.equal(Test_Labels, 0))
    
    print(TP, FP, FN, TN)

    Sens.append(TP / (TP + FN)) 
    TFPo.append(FP / (FP + TN))
    
fig, ax = plt.subplots()
ax.scatter(TFPo, Sens)
ax.set_xlabel('TFPo')
ax.set_ylabel('Sens')
plt.show()


# In[83]:


min(TFPo)


# Na etapa de teste foi utilizado o melhor vetor de parâmetros obtido na etapa de validação. Com ele os seguintes valores de métricas foram alcançados:
# 
# - Acurácia Balanceada  = **0.73**
# - Acurácia             = **0.76**
# - F-medida             = **0.83**

# In[270]:


# Knn para a validação


 
def KNN(K, Fi_Set, Fi_Train):
    Label_Calcs_Bruto = np.empty((0,1))
    Label_Calcs = np.empty((0,1))
    for Novo_Padrao in Fi_Set[:, 1:len(Fi_Set[0])]:
        Distancias = np.empty((0, 2))
        for Padrao, Label in zip(Fi_Train[:, 1:len(Fi_Train[0])], Train_Labels):
            Distancia = [np.linalg.norm(Novo_Padrao - Padrao), Label[0]]
            Distancias = np.vstack((Distancias, Distancia))
        Distancias = np.sort(Distancias.view('f8, f8'), order=['f0'], axis=0)
        Distancias = Distancias[0:K]
        Distancias = Distancias.view(dtype = np.float64)
        Label_Calc = np.sum(Distancias, axis = 0)[1] / K
        Label_Calcs_Bruto = np.vstack((Label_Calcs_Bruto, Label_Calc))
        print(Label_Calcs_Bruto)
        if Label_Calc >= .5:
            Label_Calc = 1
        else:
            Label_Calc = 0
        Label_Calcs = np.vstack((Label_Calcs, Label_Calc))
    return Label_Calcs, Label_Calcs_Bruto

Ks = []
BAs = []
for K in range(1, 20, 1):    
    
    Label_Calcs, Label_Calcs_Bruto = KNN(K, Fi_Val, Fi_Train)

    TP = np.sum(np.equal(Label_Calcs, 1) & np.equal(Val_Labels, 1))
    FP = np.sum(np.equal(Label_Calcs, 1) & np.equal(Val_Labels, 0))
    FN = np.sum(np.equal(Label_Calcs, 0) & np.equal(Val_Labels, 1))
    TN = np.sum(np.equal(Label_Calcs, 0) & np.equal(Val_Labels, 0))


    Matriz_Conf = np.array([[TP, FN], 
                            [FP, TN]])    

    Espe = TN / (TN + FP)
    Sens = TP / (TP + FN)
    Prec = TP / (TP + FP)
    TFPo = FP / (TN + FP)

    BA = (Espe + Sens) / 2
    Acc = (TP + TN) / (TP + TN + FP + FN)
    Fm = (2 * Sens * Prec) / (Sens + Prec)
    
    BAs.append(BA)
    Ks.append(K)

fig, ax = plt.subplots()
ax.scatter(Ks, BAs)
ax.set_xlabel('Ks')
ax.set_ylabel('BAs')
plt.show()

        


# In[ ]:


#Curva ROC

Sens = []
TFPo = []
Thresholds = np.arange(0, 1, 0.0001)

for Threshold in Thresholds:
    
    Label_Calc_ROC = np.empty((0,1))
    for Label_Calc_Bruto in Label_Calcs_Bruto:
        if Label_Calc_Bruto >= Threshold:
            Label_Calc_Bruto = 1
        else:
            Label_Calc_Bruto = 0
        Label_Calc_ROC = np.vstack((Label_Calc_ROC, Label_Calc_Bruto))
    #print(Label_Calc_ROC)

    TP = np.sum(np.equal(Label_Calc_ROC, 1) & np.equal(Test_Labels, 1))
    FP = np.sum(np.equal(Label_Calc_ROC, 1) & np.equal(Test_Labels, 0))
    FN = np.sum(np.equal(Label_Calc_ROC, 0) & np.equal(Test_Labels, 1))
    TN = np.sum(np.equal(Label_Calc_ROC, 0) & np.equal(Test_Labels, 0))
    
    #print(TP, FP, FN, TN)

    Sens.append(TP / (TP + FN)) 
    TFPo.append(FP / (FP + TN))
    
fig, ax = plt.subplots()
ax.scatter(TFPo, Sens)
ax.set_xlabel('Sens')
ax.set_ylabel('TFPo')
plt.show()


# In[271]:


# Knn para o teste com K = 1
K = 20
Label_Calcs, Label_Calcs_Bruto = KNN(K, Fi_Test, Fi_Train)

TP = np.sum(np.equal(Label_Calcs, 1) & np.equal(Test_Labels, 1))
FP = np.sum(np.equal(Label_Calcs, 1) & np.equal(Test_Labels, 0))
FN = np.sum(np.equal(Label_Calcs, 0) & np.equal(Test_Labels, 1))
TN = np.sum(np.equal(Label_Calcs, 0) & np.equal(Test_Labels, 0))


Matriz_Conf = np.array([[TP, FN], 
                        [FP, TN]])    

Espe = TN / (TN + FP)
Sens = TP / (TP + FN)
Prec = TP / (TP + FP)
TFPo = FP / (TN + FP)

BA = (Espe + Sens) / 2
Acc = (TP + TN) / (TP + TN + FP + FN)
Fm = (2 * Sens * Prec) / (Sens + Prec)


# In[274]:


#Curva ROC

Sens = []
TFPo = []
Thresholds = np.arange(0, 1, 0.0001)

for Threshold in Thresholds:
    
    Label_Calc_ROC = np.empty((0,1))
    for Label_Calc_Bruto in Label_Calcs_Bruto:
        if Label_Calc_Bruto >= Threshold:
            Label_Calc_Bruto = 1
        else:
            Label_Calc_Bruto = 0
        Label_Calc_ROC = np.vstack((Label_Calc_ROC, Label_Calc_Bruto))
    #print(Label_Calc_ROC)

    TP = np.sum(np.equal(Label_Calc_ROC, 1) & np.equal(Test_Labels, 1))
    FP = np.sum(np.equal(Label_Calc_ROC, 1) & np.equal(Test_Labels, 0))
    FN = np.sum(np.equal(Label_Calc_ROC, 0) & np.equal(Test_Labels, 1))
    TN = np.sum(np.equal(Label_Calc_ROC, 0) & np.equal(Test_Labels, 0))
    
    #print(TP, FP, FN, TN)

    Sens.append(TP / (TP + FN)) 
    TFPo.append(FP / (FP + TN))
    
fig, ax = plt.subplots()
ax.scatter(TFPo, Sens)
ax.set_xlabel('Sens')
ax.set_ylabel('TFPo')
plt.show()


# In[269]:


Label_Calcs


# In[21]:


w = [5]
w = 25*w
w = np.array(w)


# In[91]:


Fm


# In[207]:


w.view(dtype = 'float16')


# In[188]:


thresholds = np.logspace(-100, 0, num=100-0+1,base=10,dtype='float64')
for threshold in thresholds:
    if threshold > np.array([1e-50]):
        print('maior')
    else:
        print('menor')
    


# In[107]:


np.geq


# In[181]:


np.logspace(-100, 0, num=100-0+1,base=10,dtype='float64')


# In[211]:


fig, ax = plt.subplots()
ax.scatter(np.logspace(-100, 0, num=len(Label_Calcs_Bruto),base=10,dtype='float128'), Label_Calcs_Bruto)
#ax.set_xlabel('TFPo')
#ax.set_ylabel('Sens')
plt.show()


# Se ele chutar todos como 1 -> TP / TP + FN = 113 / 113 + 0 = 1; FP / FP + TN = 42 / 42 + 0 = 1
# Se ele chutar todos como 0 -> TP / TP + FN = 0 / 0 + 113 = 0; FP / FP + TN = 0 / 0 + 72 = 0 
