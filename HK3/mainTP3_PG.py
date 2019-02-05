### PLASSIER Vincent : Master M2 MVA 2018/2019 - Reinforcement Learning - HWK 3
import numpy as np
import matplotlib.pyplot as plt
import lqg1d
import utils
import scipy.stats as stats
import bisect
from tqdm import tqdm

plt.show()
plt.ion()


## All the function use :
class Constant(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def update(self, p):
        return self.learning_rate * p
class Annealing:
    def __init__(self, alpha=1, function=lambda x: 1/x):
        self.alpha, self.function, self.iteration = alpha, function, 1
    def reset(self):
        self.iteration = 1
    def update(self, p):
        step = self.alpha * self.function(self.iteration)
        self.iteration += 1
        return step * p
class Adam:
    def __init__(self, alpha=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha, self.beta1, self.beta2, self.epsilon = alpha, beta1, beta2, epsilon
    def reset(self):
        self.i, self.n, self.u, self.f, self.hat_n, self.hat_u = 0, 0, 0, 0, 0, 0
    def update(self, grad):
        self.i += 1
        self.f = grad
        self.n = self.beta1 * self.n + (1 - self.beta1) * self.f
        self.u = self.beta2 * self.u + (1 - self.beta2) * self.f ** 2
        self.hat_n = self.n / (1 - self.beta1 ** self.i)
        self.hat_u = self.u / (1 - self.beta2 ** self.i)
        return (self.alpha * self.hat_n) / (np.sqrt(self.hat_u) + self.epsilon)
        
class policy_gaussian:
    def __init__(self, theta):
        self.theta = theta
        self.mu = lambda s: self.theta * s
        self.mu_prime = lambda s: s
        self.sigma = lambda s: 0.4
        self.theta_records = []
    def log_gradient(self, s, a):
        sigma_sqr = self.sigma(s) ** 2
        return ((a - self.mu(s)) / sigma_sqr) * self.mu_prime(s)
    def draw_action(self, s):
        mu = self.mu(s)
        sigma = self.sigma(s)
        a = np.random.normal(mu, sigma)
        return a

def gradient_J(policy, state, action, reward, gamma):
    T = action.shape[0]
    grad = 0
    for t in range(T):
        n = reward[t:].shape[0]
        r = np.sum(reward[t:]*(gamma**np.arange(n)))
        grad += r*policy.log_gradient(state[t], action[t])
    return grad

def Explo_bonus(states, actions, beta, S_space, A_space):
    nb_pts = len(S_space)
    N = np.zeros((100, nb_pts)) 
    T = states.shape[0]
    exp_bon = np.zeros(T)
    for t in range(0, T):
        s = states[t]
        a = actions[t]
        x = bisect.bisect(S_space,s)
        y = bisect.bisect(S_space,a)
        N[x,y] += 1
        exp_bon[t] = beta * np.sqrt(1 / N[x,y])
    return exp_bon

#####################################################
# Define the environment and the policy
#####################################################
env = lqg1d.LQG1D(initial_state_type='random')

policy = policy_gaussian

#####################################################
# Experiments parameters
#####################################################
# We will collect N trajectories per iteration
N = 60
# Each trajectory will have at most T time steps
T = 100
# Number of policy parameters updates
n_itr = 100
# Number of epochs
epochs = 5
# Set the discount factor for the problem
discount = 0.9
# Learning rate for the gradient update
learning_rate = 0.00001


## Ex 1 :
stepper = Adam() # we choose the Adam step
list_mean_parameters, list_avg_returns= [], []

for e in tqdm(range(epochs), desc="Simulating {}".format('random')):
    stepper.reset()
    theta = 0 # we initialize theta
    avg_return, mean_parameters = [], [theta]
    for _ in range(n_itr):
        pi = policy(theta)
        paths = utils.collect_episodes(env, policy=pi, horizon=T, n_episodes=N)
        l = len(paths)
        avg = np.sum([paths[n]["rewards"] for n in range(l)])/l
        grad_J = np.sum([gradient_J(pi,paths[n]["states"][:,0], paths[n]["actions"][:,0],paths[n]["rewards"], discount) for n in range(N)]) / N
        theta += stepper.update(grad_J)
        avg_return.append(avg)
        mean_parameters.append(theta)
    list_avg_returns.append(avg_return)
    list_mean_parameters.append(mean_parameters)
    

list_avg_returns = np.array(list_avg_returns)
list_mean_parameters = np.array(list_mean_parameters)

avg_returns = np.mean(list_avg_returns,axis=0)
mean_parameter = np.mean(list_mean_parameters,axis=0)

std_returns = np.std(list_avg_returns,axis=0)
std_parameter = np.std(list_mean_parameters,axis=0)


# We plot the average return
plt.figure(1)
plt.subplot(121)
y1 = avg_returns + 1.96 * std_returns
y2 = avg_returns - 1.96 * std_returns
x = np.arange(0, avg_returns.shape[0])
plt.fill_between(x, y1, y2, alpha=.1)
plt.plot(avg_returns, label="Reward")
plt.ylabel('Reward')
plt.xlabel('Rounds')
plt.legend()

# We plot the parameter theta in function of the number of policy updates
plt.subplot(122)
y1 = mean_parameter + 1.96 * std_parameter
y2 = mean_parameter - 1.96 * std_parameter
x = np.arange(0, mean_parameter.shape[0])
plt.fill_between(x, y1, y2, alpha=.1)
plt.plot(mean_parameter, label="theta")
plt.ylabel('theta')
plt.xlabel('Rounds')
plt.legend()


## Ex 2 :
nb_pts = 100 # number of bins
S_space = np.linspace(-40,40,nb_pts)
A_space = np.linspace(-40,40,nb_pts)
beta = 10

stepper = Adam() # we choose the Adam step
list_mean_parameters, list_avg_returns= [], []

for e in tqdm(range(epochs), desc="Simulating {}".format('random')):
    stepper.reset()
    theta = 0 # we initialize theta
    avg_return, mean_parameters = [], [theta]
    for _ in range(n_itr):
        pi = policy(theta) 
        paths = utils.collect_episodes(env, policy=pi, horizon=T, n_episodes=N)
        l = len(paths)
        avg = np.sum([paths[i]["rewards"] for i in range(l)])/l
        grad_J = 0 # to calculate the gradient
        for i in range(0, N):
            exp_bon = Explo_bonus(paths[i]["states"], paths[i]["actions"], beta, S_space, A_space)
            grad_traj = gradient_J(pi,paths[i]["states"][:,0], paths[i]["actions"][:,0], paths[i]["rewards"] + exp_bon, discount)
            grad_J += grad_traj/N
        theta += stepper.update(grad_J)
        avg_return.append(avg)
        mean_parameters.append(theta)
    list_avg_returns.append(avg_return)
    list_mean_parameters.append(mean_parameters)

    
list_avg_returns = np.array(list_avg_returns)
list_mean_parameters = np.array(list_mean_parameters)

avg_returns = np.mean(list_avg_returns,axis=0)
mean_parameter = np.mean(list_mean_parameters,axis=0)

std_returns = np.std(list_avg_returns,axis=0)
std_parameter = np.std(list_mean_parameters,axis=0)


# We plot the average return
plt.figure(2)
plt.subplot(121)
y1 = avg_returns + 1.96 * std_returns
y2 = avg_returns - 1.96 * std_returns
x = np.arange(0, avg_returns.shape[0])
plt.fill_between(x, y1, y2, alpha=.1)
plt.plot(avg_returns, label="Reward")
plt.ylabel('Reward')
plt.xlabel('Rounds')
plt.legend()

# We plot the parameter theta in function of the number of policy updates
plt.subplot(122)
y1 = mean_parameter + 1.96 * std_parameter
y2 = mean_parameter - 1.96 * std_parameter
x = np.arange(0, mean_parameter.shape[0])
plt.fill_between(x, y1, y2, alpha=.1)
plt.plot(mean_parameter, label="theta")
plt.ylabel('theta')
plt.xlabel('Rounds')
plt.legend()