from models import *
import tensorflow as tf
from replay_buffer import ReplayBuffer
import numpy as np

class DDPGagent:
    def __init__(self, num_actions, num_observations ,actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=50000):
        #Params
        self.num_actions = num_actions
        self.num_observations = num_observations
        self.gamma = gamma
        self.tau = tau
        
        #Network
        self.actor = Actor(num_observations,num_actions)
        self.actor_target = Actor(num_observations,num_actions)
        self.actor_target.set_weights(self.actor.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_learning_rate)
        self.critic = Critic(num_observations,num_actions)
        self.critic_target = Critic(num_observations,num_actions)
        self.critic_target.set_weights(self.critic.get_weights())
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)
        
        #Memory
        self.memory = ReplayBuffer(max_memory_size)
    
    def get_action(self, state):
        state = np.float32(np.expand_dims(state, axis=0))
        action = self.actor(state)[0]#Convert to float32
        return(action)
    
    #@tf.function
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        next_states = tf.convert_to_tensor(next_states)
        
        #Critic
        with tf.GradientTape() as tape:
            Qvals = self.critic([states, actions])
            next_actions = self.actor_target(next_states)
            next_Q = self.critic_target([next_states, next_actions])
            Qprime = rewards + self.gamma * next_Q
            critic_loss = tf.keras.losses.mean_squared_error(Qvals, Qprime)
        gradients_of_critic = tape.gradient(critic_loss, self.critic.trainable_variables)   
        self.critic_optimizer.apply_gradients(zip(gradients_of_critic, self.critic.trainable_variables))
        
        #Actor
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            q_gradients = -self.critic([states, actions])
        parameter_gradients = tape.gradient(q_gradients, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(parameter_gradients, self.actor.trainable_variables))
        
        #Update target Critic
        target_param = self.critic_target.get_weights()
        param = self.critic.get_weights()
        for layer in range(0, len(target_param)):
            target_param[layer] = self.tau * param[layer] + (1.0 - self.tau) * target_param[layer]
        self.critic_target.set_weights(target_param)
        
        #update target Actor
        target_param = self.actor_target.get_weights()
        param = self.actor.get_weights()
        for layer in range(0, len(target_param)):
            target_param[layer] = self.tau * param[layer] + (1.0 - self.tau) * target_param[layer]
        self.actor_target.set_weights(target_param)
        
        
