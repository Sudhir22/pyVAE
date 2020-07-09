"""
Code referenced from https://www.geeksforgeeks.org/ml-autoencoder-with-tensorflow-2-0/

"""
from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function 

import tensorflow as tf 
print(tf.__version__) 
import numpy as np 
import sklearn.preprocessing as prep 
import tensorflow.keras.layers as layers 

def standard_scale(X_train, X_test): 
    preprocessor = prep.StandardScaler().fit(X_train) 
    X_train = preprocessor.transform(X_train) 
    X_test = preprocessor.transform(X_test) 
    return X_train, X_test 

def get_random_block_from_data(data, batch_size): 
    start_index = np.random.randint(0, len(data) - batch_size) 
    return data[start_index:(start_index + batch_size)] 


class Encoder(tf.keras.layers.Layer): 
    '''Encodes a digit from the MNIST dataset'''
    
    def __init__(self,n_dims,name ='encoder',**kwargs): 
        super(Encoder, self).__init__(name = name, **kwargs) 
        self.n_dims = n_dims 
        self.n_layers = len(n_dims)
        self.encode_layer = list()
        for x in range(0,self.n_layers):
            self.encode_layer.append(layers.Dense(n_dims[x], activation ='relu'))

    @tf.function
    def call(self, inputs): 
        for x in range(0,self.n_layers):
            inputs = self.encode_layer[x](inputs)
        return inputs 

class Decoder(tf.keras.layers.Layer): 
    '''Decodes a digit from the MNIST dataset'''

    def __init__(self,n_dims,name ='decoder',**kwargs): 
        super(Decoder, self).__init__(name = name, **kwargs) 
        self.n_dims = n_dims 
        self.n_layers = len(n_dims)
        self.decode_middle = list()
        for x in range(0,self.n_layers-1):
            self.decode_middle.append(layers.Dense(n_dims[x], activation ='relu'))
        self.recon_layer = layers.Dense(n_dims[len(n_dims)-1], activation ='sigmoid') 

    @tf.function
    def call(self, inputs):
        for x in range(0,self.n_layers-1):
            inputs = self.decode_middle[x](inputs) 
        return self.recon_layer(inputs) 



class Autoencoder(tf.keras.Model): 
    '''Vanilla Autoencoder for MNIST digits'''
    
    def __init__(self,n_dims_encoder =[200],n_dims_decoder = [392, 784],name ='autoencoder',**kwargs): 
        super(Autoencoder, self).__init__(name = name, **kwargs) 
        self.encoder = Encoder(n_dims_encoder) 
        self.decoder = Decoder(n_dims_decoder) 
    
    @tf.function
    def call(self, inputs):
        x = self.encoder(inputs)
        return self.decoder(x)


mnist = tf.keras.datasets.mnist 

(X_train, _), (X_test, _) = mnist.load_data() 
X_train = tf.cast(np.reshape(X_train, (X_train.shape[0],X_train.shape[1] * X_train.shape[2])), tf.float64) 
X_test = tf.cast(np.reshape(X_test,(X_test.shape[0],X_test.shape[1] * X_test.shape[2])), tf.float64) 

X_train, X_test = standard_scale(X_train, X_test) 


train_data = tf.data.Dataset.from_tensor_slices( 
		X_train).batch(128).shuffle(buffer_size = 1024) 
test_data = tf.data.Dataset.from_tensor_slices( 
		X_test).batch(128).shuffle(buffer_size = 512) 

n_samples = int(len(X_train) + len(X_test)) 
training_epochs = 20
batch_size = 128
display_step = 1

optimizer = tf.optimizers.Adam(learning_rate = 0.01) 
mse_loss = tf.keras.losses.MeanSquaredError() 
loss_metric = tf.keras.metrics.Mean() 

ae = Autoencoder([200],[392, 784]) 
ae.compile(optimizer = tf.optimizers.Adam(0.01), 
		loss ='categorical_crossentropy') 
ae.fit(X_train, X_train, batch_size = 64, epochs = 10) 

