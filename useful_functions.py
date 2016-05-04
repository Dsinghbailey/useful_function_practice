import numpy as np
import random




# softmax function
# gradient-log-normalizer of the categorical probability distribution.
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# Q-Learning Class
# Generalized from smartcab project
# TODO: test and debug
'''
This class needs a custom update function
Sample update function:

actions = ('forward', 'left', 'right', None)
agent = QLearningAgent(actions, .5, .5, .1)
self.state1 = (inputs['light'],
                      self.next_waypoint)

action = self.choose_action(self.state)

# Execute action and get reward
reward = self.env.act(self, action)

self.state2 = (inputs['light'],
                      self.next_waypoint)

self.learn(self.state, action, reward, self.p_state)
'''
class QLearningAgent():

    def __init__(self, actions, alpha, gamma, epsilon):
        self.actions = actions
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount
        self.epsilon = epsilon # exploration
        self.q = {}

    def qlearn_getQ(self, state, action):
                return self.q.get((state, action), 0.0)

    def qlearn_choose_action(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)
            action = self.actions[i]
        return action

    def qlearn_learnQ(self, state, action, reward, value):
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)
