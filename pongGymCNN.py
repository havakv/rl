# Modified scrip from http://karpathy.github.io/2016/05/31/rl/
""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
# import cPickle as pickle
import pickle
import gym
from keras.layers import Dense, Input, Flatten, Convolution2D
from keras.models import Model, load_model
from keras.optimizers import RMSprop
import keras.backend as K

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
# learning_rate = 1e-4
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False
resume = True # resume from previous checkpoint?
render = True

# model initialization
D = 80 # input dimensionality: 80x80 grid
model_file_name = 'pong_gym_cnn.h5'

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,:1] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    # return I.astype(np.float).ravel()
    return I.astype(np.float)

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def store_all_at_iteration(prefix='', ostfix=''):
    """Store all environment variables."""
    pickle.dump(observation, open('observation.p', 'wb'))
    pickle.dump(reward, open('reward.p', 'wb'))
    pickle.dump(done, open('done.p', 'wb'))
    pickle.dump(info, open('info.p', 'wb'))

def get_dense_model():
    """Make keras model"""
    inp = Input(shape=(D,))
    # flat = Flatten()(inp)
    h = Dense(H, activation='relu')(inp)
    out = Dense(1, activation='sigmoid')(h)
    model = Model(inp, out)
    # optim = RMSprop(learning_rate, decay=decay_rate)
    # optim = RMSprop(learning_rate)
    # model.compile(optim, 'binary_crossentropy')
    model.compile('adam', 'binary_crossentropy')
    return model

def get_cnn_model():
    """Make keras cnn mode."""
    inp = Input(shape=(D, D, 1))
    x = Convolution2D(16, 8, 8, subsample=(4, 4), activation='relu')(inp)
    x = Convolution2D(32, 4, 4, subsample=(2, 2), activation='relu')(x)
    # x = Convolution2D(, 3, 3, subsample=(1, 1), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    optim = RMSprop(learning_rate)
    model.compile(optim, 'binary_crossentropy')
    # model.compile('adam', 'binary_crossentropy')
    return model

if resume:
    model = load_model(model_file_name)
else:
    # model_keras = get_dense_model()
    model = get_cnn_model()

def main():
    env = gym.make("Pong-v0")
    observation = env.reset()
    prev_x = None # used in computing the difference frame
    xs,drs,ys = [],[],[]
    running_reward = None
    reward_sum = 0
    episode_number = 0
    while True:
        if render: env.render()
    
        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros((D, D, 1))
        prev_x = cur_x
    
        # forward the policy network and sample an action from the returned probability
        # aprob = model_keras.predict(x.reshape((1, -1)))
        aprob = model.predict(np.stack([x]))
        action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
    
        # record various intermediates (needed later for backprop)
        # xs.append(x.reshape((1, -1))) # observation
        xs.append(x) # observation
        y = 1 if action == 2 else 0 # a "fake label"
        ys.append(y)
    
        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += reward
    
        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
    
        if done: # an episode finished
            episode_number += 1
            # print(episode_number)

            if episode_number % batch_size == 0:
                print('Updating weights...')
                # stack together all inputs, hidden states, action gradients, and rewards for this episode
                epx = np.stack(xs)
                epy = np.vstack(ys)
                epr = np.vstack(drs)
                xs,drs,ys = [],[],[] # reset array memory
    
                # compute the discounted reward backwards through time
                discounted_epr = discount_rewards(epr)
                # standardize the rewards to be unit normal (helps control the gradient estimator variance)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
    
                #-----------------
                # keras stuff
                # update our model weights
                model.fit(epx, epy, batch_size=512, nb_epoch=1, verbose=0, 
                        sample_weight=discounted_epr.reshape((-1,)))
    
    
                #-----------------
    
            # boring book-keeping
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            # if episode_number % 100 == 0: pickle.dump(model, open(weight_file_name, 'wb'))
            if episode_number % 10 == 0: model.save(model_file_name)
            reward_sum = 0
            observation = env.reset() # reset env
            prev_x = None
    
        if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
            print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))

if __name__ == '__main__':
    main()
