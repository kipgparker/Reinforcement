import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

def Actor(obs_dim,action_dim):
    obs = Input(obs_dim)
    l2 = Dense(128)(obs)
    #l3 = Dense(128)(l2)
    outputs = Dense(action_dim)(l2)
    
    model = Model(inputs=[obs], outputs=[outputs])
    return model

def Critic(obs_dim,action_dim):
    obs = Input(obs_dim)
    action = Input(action_dim)
    l2 = Dense(128)(obs)
    cat = concatenate([l2,action])
    l3 = Dense(8)(cat)
    #l4 = Dense(300)(l3)
    outputs = Dense(1)(l3)
    
    model = Model(inputs=[obs,action], outputs=[outputs])
    return model