import numpy as np
import utils
import scipy.stats as stats
import bisect

class ConstantStep(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def update(self, gt):
        return self.learning_rate * gt
        
class AnnealingStep:
    def __init__(self, alpha, function = lambda x: np.ln(x)/x):
        self.alpha = alpha
        self.function = function
        self.iteration = 1
    def update(self, gt):
        step = self.alpha * self.function(self.iteration)
        self.iteration += 1
        return step * gt
        
class AdamStep:
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = 0
        self.v = 0
        self.g = 0
        self.hat_m = 0
        self.hat_v = 0
    def update(self, grad):
        self.t += 1
        self.g = grad
        self.m = self.beta1 * self.m + (1 - self.beta1) * self.g
        self.v = self.beta2 * self.v + (1 - self.beta2) * self.g ** 2
        self.hat_m = self.m / (1 - self.beta1 ** self.t)
        self.hat_v = self.v / (1 - self.beta2 ** self.t)
        return (self.alpha * self.hat_m) / (np.sqrt(self.hat_v) + self.epsilon)
    def reset(self):
        self.t = 0
        self.m = 0
        self.v = 0
        self.g = 0
        self.hat_m = 0
        self.hat_v = 0

class GaussianPolicy:
    def __init__(self, theta):
        self.theta = theta
        self.mu = lambda s: self.theta * s
        self.mu_prime = lambda s: s
        self.sigma = lambda s: 0.4
        self.theta_records = []
    def draw_action(self, s):
        mu = self.mu(s)
        sigma = self.sigma(s)
        a = np.random.normal(mu, sigma)
        return a
    def get_theta(self):
        return self.theta
    def set_theta(self, new_theta):
        self.theta_records.append(self.theta)
        self.theta = new_theta
    def gradient(self, s, a):
        sigma_sqr = self.sigma(s) ** 2
        return ((a - self.mu(s)) / sigma_sqr) * self.mu_prime(s)

class Bins:
    def __init__(self, pace_s, pace_a, bounds=((-40, 40), (-40, 40))):
        self.pace_s = pace_s
        self.pace_a = pace_a
        self.bounds = bounds
        self.t = 0
        self.bins_s = np.arange(bounds[0][0], bounds[0][1], pace_s)
        self.bins_a = np.arange(bounds[1][0], bounds[1][1], pace_a)
        self.counts = dict()
    def update(self, s, a):
        i = bisect.bisect(self.bins_s, s)
        j = bisect.bisect(self.bins_a, a)
        if (i, j) in self.counts.keys():
            self.counts[(i, j)] += 1
        else:
            self.counts[(i, j)] = 1
        self.t += 1
        return self.counts[(i, j)]
    def reset(self):
        self.counts = dict()
        self.t = 0
        
def R_estimate(rewards, gamma, t):
    size = rewards[t:].shape[0]
    discounts = np.exp(np.log(gamma) * np.arange(0, size, 1))
    return np.sum(rewards[t:] * discounts)
    
def bonus_functions(states, actions, beta, pace_s=1, pace_a=1):
    bins = Bins(pace_s, pace_a)
    T = states.shape[0]
    bonuses = np.zeros(T)
    for t in range(0, T):
        s = states[t]
        a = actions[t]
        count = bins.update(s, a)
        bonuses[t] = beta * np.sqrt(1 / count)
    return bonuses
    
def trajectory_gradient(policy, states, actions, rewards, gamma):
    grad = 0
    T = actions.shape[0]
    for t in range(0, T):
        grad += policy.gradient(states[t], actions[t]) * (R_estimate(rewards, gamma, t))
    return grad
def gradient_J(policy, states, actions, rewards, gamma):
    grad = 0
    T = actions.shape[0]
    for t in range(0, T):
        size = rewards[t:].shape[0]
        discounts = gamma**np.arange(0, size, 1)
        r = np.sum(rewards[t:] * discounts)
        grad += policy.gradient(states[t], actions[t]) * r
    return grad
    
def average_return(paths):
    avg_returns = 0
    N = len(paths)
    for n in range(0, N):
        avg_returns += np.sum(paths[n]["rewards"])
    return avg_returns/N
    
class UniformPolicy:
    def __init__(self, bounds=(-40, 40), pace=1):
        self.bounds = bounds
        self.pace = pace
        self.action_space = np.arange(bounds[0], bounds[1], pace)
    def draw_action(self, s):
        return np.random.choice(self.action_space)
        
def build_Z(actions, states):
    T = actions.shape[0]
    Z = np.zeros((3, T))
    Z[0, :] = actions
    Z[1, :] = actions * states
    Z[2, :] = actions ** 2 + states ** 2
    return Z
    
def empirical_bellman(a, s, ns, r, action_space, gamma):
    ret