# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
import tensorflow as keras
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import Model
import matplotlib.pyplot as plt

# load the dataset
#dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
#X = np.concatenate((dataset[:,0:8],-dataset[:,0:8],np.ones((dataset.shape[0],1))),axis=1)
#X=dataset[:,0:8]
#y = dataset[:,8]

#scaler = preprocessing.StandardScaler().fit(X)
#X = scaler.transform(X)

# Split the remaining data to train and validation
#x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.15, shuffle=True,random_state=1000)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


np.random.seed(1000)
a=(np.array(np.random.rand(x_train.shape[0],x_train.shape[1],x_train.shape[2])>0.3)).astype(float)#0.3
#a=1*(np.array(np.random.rand(x_train.shape[0],x_train.shape[1],x_train.shape[2]))).astype(float)#0.3
#np.random.seed(1010)
#b=1*(np.array(np.random.rand(x_train.shape[0],x_train.shape[1],x_train.shape[2]))).astype(float)#0.3
#np.random.seed(1020)
#c=1*(np.array(np.random.rand(x_train.shape[0],x_train.shape[1],x_train.shape[2]))).astype(float)#0.3
x_train=np.maximum(a,x_train)
#x_train=a+x_train

#plt.imshow(x_train[0,:,:])
#plt.show()
#exit(1)

x_train=x_train.reshape((-1, 28*28))
x_test=x_test.reshape((-1, 28*28))


'''
# define the keras model
model = Sequential()
model.add(Dense(96, input_dim=8, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

'''
x_in = Input(shape=(28*28,))
x = Dense(32, activation='relu')(x_in)
x_1 = Dense(32, activation='relu')(x)
x_2 = Dense(32, activation='relu')(x_1)
x_3 = Dense(32, activation='relu')(x_2)
x_4 = Dense(32, activation='relu')(x_3)
x_5 = Dense(32, activation='relu')(x_4)
#x_6 = Dense(100, activation='relu')(x_5)
#x_7 = Dense(100, activation='relu')(x_6)
#x_8 = Dense(100, activation='relu')(x_7)
#x_9 = Dense(100, activation='relu')(x_8)
#x_10 = Dense(100, activation='relu')(x_9)
x_out = Dense(10, activation='sigmoid')(x_5)
#'''



model = Model(inputs=x_in, outputs=x_out)

# compile the keras model
optimizer = keras.optimizers.Adam()#lr=0.001#0.0001
#model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])#optimizer='adam'
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=optimizer, metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(x=x_train, y=y_train, verbose=2, epochs=150, batch_size=100, validation_data=(x_test, y_test))#1500 #epochs=150, batch_size=100#epochs=1500, batch_size=60000

# evaluate the keras model
#_, accuracy = model.evaluate(X, y)
_, accuracy = model.evaluate(x_test, y_test)
print('Acc: %.2f' % (accuracy*100))