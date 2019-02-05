### PLASSIER Vincent : Master M2 MVA 2017/2018 - Reinforcement Learning - HWK 2

import numpy as np
import arms
import matplotlib.pyplot as plt

plt.show()
plt.ion()


##
# Build your own bandit problem
def bernouilli_MAB(list_p):
    MAB = []
    for k in range(0, list_p.shape[0]):
        MAB.append(arms.ArmBernoulli(list_p[k], random_state=np.random.randint(1, 312414)))
    return MAB
    
    
# this is an example, please change the parameters or arms!
arm1 = arms.ArmBernoulli(0.3, random_state=np.random.randint(1, 312414))
arm2 = arms.ArmBernoulli(0.4, random_state=np.random.randint(1, 312414))
arm3 = arms.ArmBernoulli(0.2, random_state=np.random.randint(1, 312414))
arm4 = arms.ArmBernoulli(0.1, random_state=np.random.randint(1, 312414))

MAB = [arm1, arm2, arm3, arm4]

# bandit : set of arms

nb_arms = len(MAB)
means = [el.mean for el in MAB]

# Display the means of your bandit (to find the best)
print('means: {}'.format(means))
mu_max = np.max(means)

# Comparison of the regret on one run of the bandit algorithm
# try to run this multiple times, you should observe different results

T = 5000  # horizon


## 1.
def UCB1(T,MAB,rho=2):
    nb_arms = len(MAB)
    rew, draw= np.zeros(T), np.zeros(T)
    N=np.zeros(nb_arms) # number of draws of arms up to time t
    S=np.zeros_like(N) # sum of rewards gathered up to time t
    for t in range(T):
        if t<nb_arms:
            a_t=t
        else:
            a_t = np.argmax(S/N + rho**np.sqrt(np.log(t)/(2*N))) # algorithm picks the action
        rew[t] = MAB[a_t].sample() # get the reward
        draw[t] = a_t
        # update algorithm
        N[a_t]+=1
        S[a_t]+=rew[t]
    return np.asarray(rew), np.asarray(draw)


def TS(T,MAB):
    nb_arms = len(MAB)
    rew, draw= np.zeros(T), np.zeros(T)
    N=np.zeros(nb_arms) # number of draws of arms up to time t
    S=np.zeros_like(N) # sum of rewards gathered up to time t
    tau=np.zeros(nb_arms)
    for t in range(T):
        for a in range(nb_arms):
            if N[a]==0:
                tau[a]=np.random.rand()
            else:
                tau[a]=np.random.beta(S[a]+1,N[a]-S[a]+1)
        a_t = np.argmax(tau) # algorithm picks the action
        rew[t] = MAB[a_t].sample() # get the reward
        draw[t] = a_t # get the acction
        # update algorithm
        N[a_t]+=1
        S[a_t]+=rew[t]
    return np.asarray(rew), np.asarray(draw)


def Eps_Greedy(T,MAB,Eps=0.1): 
    nb_arms = len(MAB)
    rew, draw = np.zeros(T), np.zeros(T)
    N = np.zeros(nb_arms) # number of draws of arms up to time t
    S = np.zeros_like(N) # sum of rewards gathered up to time t
    for t in range(T):
        eps = np.random.rand()
        if t<nb_arms or (t>nb_arms and eps<Eps):
            a_t = np.random.randint(nb_arms)
        else :
            a_t = np.argmax(S/N)
        rew[t] = MAB[a_t].sample() # get the reward
        draw[t] = a_t # get the acction
        # update algorithm
        N[a_t] += 1
        S[a_t] += rew[t]
    return np.asarray(rew), np.asarray(draw)


rew1, draws1 = UCB1(T, MAB)
reg1 = mu_max * np.arange(1, T + 1) - np.cumsum(rew1)
rew2, draws2 = TS(T, MAB)
reg2 = mu_max * np.arange(1, T + 1) - np.cumsum(rew2)
rew3, draws3 = Eps_Greedy(T, MAB)
reg3 = mu_max * np.arange(1, T + 1) - np.cumsum(rew3)

    
plt.figure(1)
plt.clf()
x = np.arange(1, T+1)
plt.plot(x, reg1, label='UCB')
plt.plot(x, reg2, label='Thompson')
plt.plot(x, reg3, label='Eps_Greedy')
plt.xlabel('Rounds')
plt.ylabel('Cumulative Regret')
plt.legend() 


## 2.
nb_it = 20

def Expected_Regret_MC(T,MAB,nb_it):
    nb_arms=len(MAB)
    means = [el.mean for el in MAB]
    mu_max = np.max(means)
    rew_UCB1, rew_TS, rew_Eps_Greedy = np.zeros(T), np.zeros(T), np.zeros(T)
    i=0
    while i<nb_it:
        i+=1
        rew_UCB1 += UCB1(T,MAB)[0]
        rew_TS += TS(T,MAB)[0]
        rew_Eps_Greedy += Eps_Greedy(T,MAB)[0]
    return  np.arange(1,T+1)*mu_max-rew_UCB1.cumsum()/nb_it, np.arange(1,T+1)*mu_max-rew_TS.cumsum()/nb_it, np.arange(1,T+1)*mu_max-rew_Eps_Greedy.cumsum()/nb_it


list_t=np.arange(1,T+1)
R=Expected_Regret_MC(T,MAB,nb_it) 

plt.figure(2)
plt.clf()
plt.plot(list_t, R[0], label='Expected regret of UCB1')
plt.plot(list_t, R[1], label='Expected regret of TS')
plt.plot(list_t, R[2], label='Expected regret of Eps_Greedy')
plt.legend()


## 3.
kl = lambda x,y : x*np.log(x/y)+(1-x)*np.log((1-x)/(1-y))

list_pbis=[el.mean for el in MAB]
p_star=max(list_pbis)
list_pbis.remove(p_star)
list_pbis=np.asarray(list_pbis)
C=np.sum((p_star-list_pbis)/kl(list_pbis,p_star)) # we calcul C
oracle = C*np.log(list_t)


plt.figure(3)
plt.clf()
plt.plot(list_t, R[0], label='Expected regret of UCB1')
plt.plot(list_t, R[1], label='Expected regret of TS')
plt.plot(list_t, R[2], label='Eps_Greedy')
plt.plot(list_t,oracle, label='Oracle') # we display
plt.legend()


## Question 1:
arm1 = arms.ArmBernoulli(0.30, random_state=np.random.randint(1, 312414))
arm2 = arms.ArmBeta(0.20, 0.30, random_state=np.random.randint(1, 312414))
arm3 = arms.ArmExp(0.25, random_state=np.random.randint(1, 312414))
arm4 = arms.ArmFinite(np.array([0.3,0.5,0.2]), np.array([0.5,0.1,0.4]), random_state=np.random.randint(1, 312414))

MAB = [arm1, arm2, arm3, arm4]


def TS_non_binarity(T,MAB):
    nb_arms = len(MAB)
    rew, draw = np.zeros(T), np.zeros(T)
    N = np.zeros(nb_arms) # number of draws of arms up to time t
    S = np.zeros_like(N) # sum of rewards gathered up to time t
    tau = np.zeros(nb_arms)
    for t in range(T):
        for a in range(nb_arms):
            if N[a] == 0:
                tau[a] = np.random.rand()
            else:
                tau[a] = np.random.beta(S[a]+1, N[a]-S[a]+1)
        a_t = np.argmax(tau) # algorithm picks the action
        rew[t] = MAB[a_t].sample() # get the reward
        draw[t] = a_t
        # update algorithm
        N[a_t] += 1
        S[a_t] += (np.random.rand()<rew[t])
    return np.asarray(rew), np.asarray(draw)


def Expected_Regret_MC_non_binarity(T,MAB,nb_it=20):
    nb_arms=len(MAB)
    means = [el.mean for el in MAB]
    mu_max = np.max(means)
    rew_UCB1, rew_TS, rew_Eps_Greedy = np.zeros(T), np.zeros(T), np.zeros(T)
    i=0
    while i<nb_it:
        i+=1
        rew_UCB1 += UCB1(T,MAB)[0]
        rew_TS += TS_non_binarity(T,MAB)[0]
        rew_Eps_Greedy += Eps_Greedy(T,MAB)[0]
    return  np.arange(1,T+1)*mu_max-rew_UCB1.cumsum()/nb_it, np.arange(1,T+1)*mu_max-rew_TS.cumsum()/nb_it, np.arange(1,T+1)*mu_max-rew_Eps_Greedy.cumsum()/nb_it


nb_it = 20
list_t = np.arange(1,T+1)
R = Expected_Regret_MC_non_binarity(T,MAB,nb_it)

plt.figure(4)
plt.clf()
plt.plot(list_t, R[0], label='Expected regret of UCB1')
plt.plot(list_t, R[1], label='Expected regret of TS')
plt.plot(list_t, R[2], label='Eps_Greedy')
plt.plot(list_t,oracle, label='Oracle') # we display
plt.legend()