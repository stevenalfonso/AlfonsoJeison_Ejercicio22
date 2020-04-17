import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.model_selection

numeros = sklearn.datasets.load_digits()
imagenes = numeros['images']  # Hay 1797 digitos representados en imagenes 8x8
n_imagenes = len(imagenes)
X = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen 
#basta hacer data.reshape((n_imagenes, 8, 8))
Y = numeros['target']
#print(np.shape(X), np.shape(Y))

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.5)
scaler = sklearn.preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n = np.arange(1,20,1)
loss = []
score_test = []
score_train = [] 
for i in n:
    mlp=sklearn.neural_network.MLPClassifier(activation='logistic',hidden_layer_sizes=(i),max_iter=2500)
    mlp.fit(X_train, Y_train)
    loss.append(mlp.loss_)
    score_test.append(sklearn.metrics.f1_score(Y_test, mlp.predict(X_test), average='macro'))
    score_train.append(sklearn.metrics.f1_score(Y_train, mlp.predict(X_train), average='macro'))

plt.figure(figsize=(15,5))
plt.suptitle('Loss and f1_score',fontsize=15)
plt.subplot(1,2,1)
#plt.title('Loss')
plt.plot(n,loss)
plt.xlabel('neurons',fontsize=15)
plt.ylabel('loss',fontsize=15)
plt.subplot(1,2,2)
#plt.title('f1_score')
plt.plot(n,score_test,label='Test')
plt.plot(n,score_train,label='Train')
plt.xlabel('neurons',fontsize=15)
plt.ylabel('f1_score',fontsize=15)
plt.legend()
plt.savefig('loss_f1.png')

#Se escogió el número de neuronas de acuerdo a la grafica anterior dado que en este punto el f1_score se estabiliza
n = 6
mlp=sklearn.neural_network.MLPClassifier(activation='logistic',hidden_layer_sizes=(n),max_iter=2500)
mlp.fit(X_train, Y_train)
loss = mlp.loss_
f1 = sklearn.metrics.f1_score(Y_test, mlp.predict(X_test), average='macro')
#sklearn.metrics.plot_confusion_matrix(mlp, X_test, Y_test)
plt.figure(figsize=(10,5))
plt.suptitle('Loss = {:.5f} f1_score =  {:.5f}'.format(loss, f1),fontsize=15)
for i in range(n):
    plt.subplot(2,3,i+1)
    scale = np.max(mlp.coefs_[0])
    plt.title('Neuron ' + str(i))
    plt.imshow(mlp.coefs_[0][:,i].reshape(8,8), cmap=plt.cm.RdBu,vmin=-scale, vmax=scale)
plt.savefig('neuronas.png')
