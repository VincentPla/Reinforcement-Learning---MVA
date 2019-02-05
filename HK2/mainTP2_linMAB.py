### PLASSIER Vincent : Master M2 MVA 2017/2018 - Reinforcement Learning - HWK 2

import numpy as np
from linearmab_models import ToyLinearModel, ColdStartMovieLensModel
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.show()
plt.ion()


random_state = np.random.randint(0, 24532523)
# model = ToyLinearModel(
#     n_features=8,
#     n_actions=20,
#     random_state=random_state,
#     noise=0.1)

model = ColdStartMovieLensModel(
    random_state=random_state,
    noise=0.1
)

n_a = model.n_actions
d = model.n_features

T = 1000


nb_simu = 25


##################################################################
# - Random
##################################################################

regret = np.zeros((nb_simu, T))
norm_dist = np.zeros((nb_simu, T))

N = np.zeros(n_a) # number of draws of arms up to time t
S = np.zeros_like(N) # sum of rewards gathered up to time t
A = .001*np.eye(d) # aim to find theta_hat
b = np.zeros(d) # aim to find theta_hat

for k in tqdm(range(nb_simu), desc="Simulating {}".format('Random')):

    for t in range(T):
        a_t = np.random.randint(n_a)  # algorithm picks the action
        r_t = model.reward(a_t) # get the reward
        
        # update algorithm
        N[a_t] += 1
        S[a_t] += r_t
        theta_hat = np.linalg.solve(A,b)
        x = model.features[a_t]
        A += np.tensordot(x,x,0)
        b += r_t*x

        # store regret
        regret[k, t] = model.best_arm_reward() - r_t
        norm_dist[k, t] = np.linalg.norm(theta_hat - model.real_theta, 2)

# compute average (over sim) of the algorithm performance and plot it
mean_norms = np.mean(norm_dist,axis=0)
mean_regret = np.mean(regret,axis=0)

plt.figure(1)
plt.subplot(121)
plt.plot(mean_norms, label='Random')
plt.ylabel('d(theta, theta_hat)')
plt.xlabel('Rounds')
plt.legend()

plt.subplot(122)
plt.plot(mean_regret.cumsum(), label='Random')
plt.ylabel('Cumulative Regret')
plt.xlabel('Rounds')
plt.legend()

##################################################################
# - Linear UCB
##################################################################

regret = np.zeros((nb_simu, T))
norm_dist = np.zeros((nb_simu, T))

N = np.zeros(n_a) # number of draws of arms up to time t
S = np.zeros_like(N) # sum of rewards gathered up to time t
A = .001*np.eye(d) # aim to find theta_hat
b = np.zeros(d) # aim to find theta_hat
alpha = 1

for k in tqdm(range(nb_simu), desc="Simulating {}".format('Linear UCB')):

    for t in range(T):
        if t<n_a:
            a_t = t
        else:
            a_t = np.argmax(np.dot(model.features,theta_hat)+beta) # algorithm picks the action
        r_t = model.reward(a_t) # get the reward

        # update algorithm
        N[a_t] += 1
        S[a_t] += r_t
        A_inv = np.linalg.inv(A)
        theta_hat = np.dot(A_inv,b)
        x = model.features[a_t]
        beta = alpha*np.sqrt(np.dot(x,A_inv.dot(x)))
        A += np.tensordot(x,x,0)
        b += r_t*x
        
        # store regret
        regret[k, t] = model.best_arm_reward() - r_t
        norm_dist[k, t] = np.linalg.norm(theta_hat - model.real_theta, 2)

# compute average (over sim) of the algorithm performance and plot it
mean_norms = np.mean(norm_dist,axis=0)
mean_regret = np.mean(regret,axis=0)

plt.figure(1)
plt.subplot(121)
plt.plot(mean_norms, label='Linear UCB')
plt.ylabel('d(theta, theta_hat)')
plt.xlabel('Rounds')
plt.legend()

plt.subplot(122)
plt.plot(mean_regret.cumsum(), label='Linear UCB')
plt.ylabel('Cumulative Regret')
plt.xlabel('Rounds')
plt.legend()

##################################################################
# - Eps Greedy
##################################################################

regret = np.zeros((nb_simu, T))
norm_dist = np.zeros((nb_simu, T))

N = np.zeros(n_a) # number of draws of arms up to time t
S = np.zeros_like(N) # sum of rewards gathered up to time t
A = .001*np.eye(d) # aim to find theta_hat
b = np.zeros(d) # aim to find theta_hat
Eps = 0.1

for k in tqdm(range(nb_simu), desc="Simulating {}".format('Eps Greedy')):

    for t in range(T):
        eps=np.random.rand()
        if t<n_a or (t>n_a and eps<Eps):
            a_t=np.random.randint(n_a)
        else :
            a_t = np.argmax(S/N)
        r_t = model.reward(a_t) # get the reward

        # update algorithm
        N[a_t]+=1
        S[a_t]+=r_t
        theta_hat = np.linalg.solve(A,b)
        x = model.features[a_t]
        A += np.tensordot(x,x,0)
        b += r_t*x

        # store regret
        regret[k, t] = model.best_arm_reward() - r_t
        norm_dist[k, t] = np.linalg.norm(theta_hat - model.real_theta, 2)

# compute average (over sim) of the algorithm performance and plot it
mean_norms = np.mean(norm_dist,axis=0)
mean_regret = np.mean(regret,axis=0)

plt.figure(1)
plt.subplot(121)
plt.plot(mean_norms, label='Eps Greedy')
plt.ylabel('d(theta, theta_hat)')
plt.xlabel('Rounds')
plt.legend()

plt.subplot(122)
plt.plot(mean_regret.cumsum(), label='Eps Greedy')
plt.ylabel('Cumulative Regret')
plt.xlabel('Rounds')
plt.legend()