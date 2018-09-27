import numpy as np
import tensorflow as tf
from actor import ActorNetwork
from critic import CriticNetwork
from robots import gym_environment
from collections import deque

# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE =  0.01
max_episode = 1000

def actor_critic(epochs=1000, GAMMA = 0.99, train_indicator=True, render=False, temp=False):
    with tf.Session() as sess:
        
        
        # define objects
        # the gym environment is wrapped in a class. this way of working allows portability with other robots in the lab & makes the main very clear
        robot = gym_environment('FrozenLakeNonskid8x8-v0', False, render, temp) 
        actor = ActorNetwork(sess, robot.state_dim, robot.action_dim, ACTOR_LEARNING_RATE)
        critic = CriticNetwork(sess, robot.state_dim, CRITIC_LEARNING_RATE, actor.get_num_trainable_vars())
        # starting tensorflow
        sess.run(tf.global_variables_initializer())
        
        for i in range(epochs):
            # Reset the environment 
            state, done, step = robot.reset()
            ep_reward = 0
            total_reward = np.zeros(max_episode)
            total_state = deque()
            total_action = deque()
            k = 0
            while (not done) and k < max_episode:
                # Choose and take action, and observe reward
                action_prob = actor.predict(np.reshape(state,(1,robot.state_dim)))
                action = np.random.choice(np.arange(len(action_prob)), p=action_prob)
                next_state, reward, done, step = robot.update(action)
                # store episode information
                total_reward[k] = reward
                total_state.append(state)
                total_action.append(action)
                state = next_state
                k = k + 1 

            # Train 
            # get G
            for l in range(k):
                
                G = np.sum(total_reward[l:k+1])
                print(l,G)
                state = np.reshape(total_state[l], (1, robot.state_dim))
                action = np.reshape(total_action[l], (1, 1))
                
                delta = G - critic.predict(state)
                critic.train(state,delta)
                actor.train(state, action, G)

            
            
          
            # this print is usefull for debuggin  
            #print(step,'action', action, 'state', robot.uncodedstate,'r', round(reward,3), 'prob', action_prob)
            
            print('episode', i+1,'Steps', step,'Reward:',ep_reward,'goal achieved:', robot.goal,'Efficiency', round(100.*((robot.goal)/(i+1.)),0), '%' )
            
            
            
               

               
        print('*************************')
        print('now we save the model')
        critic.save_critic()
        actor.save_actor()
        print('model saved succesfuly')
        print('*************************')




if __name__ == '__main__':
    actor_critic(epochs=500)       
        