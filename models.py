import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import numpy as np

def encoder(input_size, latent_dim = 100, layers = 2):
    inputs = Input(input_size)
    
    conv = Conv2D(input_size[2], (5, 5), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    for i in range(1,layers+1):
        conv = Conv2D(np.max((np.power(2,i),input_size[2])), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv)
        conv = Dropout(0.1) (conv)
        conv = Conv2D(np.max((np.power(2,i),input_size[2])), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv)
        conv = MaxPooling2D((2, 2)) (conv)
    dense = Flatten()(conv)
    outputs = Dense(latent_dim)(dense)        
            
    model = Model(inputs = inputs, outputs = outputs)
        
    return model



def decoder(output_size, latent_dim = 100, layers = 2):

    inputs = Input((latent_dim,))
    dense = Dense(output_size[0]*2**-layers* output_size[1]*2**-layers*2**layers)(inputs)
    
    conv = Reshape((int(output_size[0]*2**-layers), int(output_size[1]*2**-layers), int(2**layers)))(dense)
    for i in range(1,layers+1):
        conv = Conv2DTranspose(np.max((np.power(2,layers-i),output_size[2])), (3, 3), strides=(2, 2), padding='same') (conv)
        conv = Conv2D(np.max((np.power(2,layers-i),output_size[2])), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv)
        conv = Dropout(0.1) (conv)
        conv = Conv2D(np.max((np.power(2,layers-i),output_size[2])), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (conv)
    conv = Conv2D(output_size[2], (3, 3), activation='sigmoid', kernel_initializer='he_normal', padding='same') (conv)


    model = Model(inputs = inputs, outputs = conv)


    return model

def agent(obs_dim, hidden_dim ,action_dim):
    
    input = Input((obs_dim,))
    x = Dense(hidden_dim)(input)
    x = Dense(hidden_dim)(x)

    advantage = Dense(hidden_dim, activation="relu")(input)
    advantage = Dense(action_dim)(advantage)
    
    advantage_norm = Lambda(lambda x: x - tf.reduce_mean(x))(advantage)
    
    value = Dense(hidden_dim, activation="relu")(input)
    value = Dense(1)(value)
    
    out = Add()([value, advantage_norm])

    model = Model(inputs=input, outputs=out)
    return model

def combine(input_size, model1, model2):
    inputs = Input(input_size) 
    first = model1()(inputs)
    second = model2()(first)
    model = Model(inputs=input, outputs=second)
    return model
    