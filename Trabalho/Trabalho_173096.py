import numpy as np
import sklearn
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score

Atributos = np.genfromtxt('Atributos.csv', delimiter=',')
Labels = np.genfromtxt('Labels.csv', delimiter=',')

#print(Atributos)

Atributos = normalize(Atributos, norm='max', axis=1)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(Atributos, Labels, test_size=0.33, random_state=42)

#print(Atributos)

MLP = MLPClassifier(hidden_layer_sizes = (100, 100),
                    max_iter = 1000,
                    tol = 0.0000000000000000001,
                    learning_rate_init = .01,
                    solver = "adam",
                    activation = "relu",
                    learning_rate = "constant",
                    verbose = 1
                    )

MLP.fit(X_Train, Y_Train) 
Preds = MLP.predict(X_Test)

#for Y, Pred in zip(Y_Test, Preds):
#    print(Y, Pred)


Preds = [np.argmax(Pred) for Pred in Preds]
Y_Test = [np.argmax(Y) for Y in Y_Test]

print(confusion_matrix(Y_Test, Preds))