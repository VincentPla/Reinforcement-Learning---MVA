### PLASSIER Vincent : Master M2 MVA 2017/2018 - Reinforcement Learning - TP 1


import numpy as np
import matplotlib.pyplot as plt

plt.ion()
plt.show()


## Q1: implementation of the MDP model
gamma=.95

X=[0,1,2] # state space
n_states=len(X) # number of possible state
A=[0,1,2] # action space
n_actions=len(A)

P=np.array([[[0.55,0.3,1],[1,0,0],[0,0,0]],[[0.45,0.7,0],[0,0.4,1],[1,0.6,0]],[[0,0,0],[0,0.6,0],[0,0.4,1]]])

r=np.array([[0,0,0.05],[0,0,0],[0,1,0.9]])

''' We guess the optimal policy is [1, 1, 2] '''

## Q2: value iteration (VI)
V1=np.zeros(n_states)

err=1
eps=(1-gamma)/(200*gamma)
list_it=[]

while err>=eps: 
    V0=V1.copy()
    for i in range(n_actions):
        V1[i]=np.max(r[i,:]+gamma*np.dot(V0,P[:,i,:]))
    list_it.append(V1.copy()) # we save the iterations
    err=np.max(abs(V1-V0))
    

def greedy_policy(V):
    pol=np.zeros(n_actions, dtype='int8')
    for i in range(n_actions):
        pol[i]=np.argmax(r[i,:]+gamma*np.dot(V,P[:,i,:]))
    return pol.astype(int)

def policy_evaluation(pol,n_states):
    r_pol=np.zeros(n_states)
    P_pol=np.zeros((n_states,n_states))
    for i,j in enumerate(pol):
        r_pol[i]=r[i,j]
        P_pol[i,:]=P[:,i,j]
    return np.linalg.solve(np.eye(n_states)-gamma*P_pol,r_pol)


pi = greedy_policy(V1)
V_star = policy_evaluation(pi,n_states)

print('pol=', pi, '\nV=', list_it[-1])

a=np.asarray(list_it)
b=np.tensordot(np.ones(len(list_it)),V_star,0)
Err=np.max(np.abs(a-b),axis=1) # list of errors between two consecutive steps


plt.figure(2)
plt.clf()
plt.plot(Err)
plt.title('Error between two consecutive steps') 


## Q3: Policy iteration (PI)
V0 = np.ones(n_states)
V1 = np.zeros(n_states) # initialization

Err=[]

while np.any(V0-V1):
    V0=V1.copy()
    pi=greedy_policy(V0)
    V1=policy_evaluation(pi,n_states)
    Err.append(np.max(np.abs(V1-V_star)))
    
print('pi=', pi, '\nV=', V1)


plt.figure(3)
plt.clf()
plt.scatter(range(len(Err)-1),Err[:-1])
plt.title('Error between two consecutive steps')


## Q4: Policy evaluation
V_pi=[0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.67106071, -0.99447514, 0.00000000, -0.82847001, -0.87691855,
        -0.93358351, -0.99447514]
        

n_states=env.n_states
n_actions=4

T_max=10**3
N_pi=10**4


pol=np.zeros(n_states, dtype='int8') # we define the policy
for state in range(n_states):
    if 0 in env.state_actions[state]:
        pol[state]=0
    else:
        pol[state]=3
        
# gui.render_policy(env, pol) # display the policy


def pi_0_func(N_pi):
    pi_0=np.zeros(n_states)
    for i in range(N_pi):
        pi_0[env.reset()]+=1
    return pi_0/N_pi


def MC_value_func(N,pi_0,pol):
    V=[]
    NN=np.int32(N*pi_0)
    NN[0]+=N-np.sum(NN) # proportionate stratification for Monte Carlo
    for state0,n in enumerate(NN):
        somme=0
        i=0
        while i<n:
            i+=1
            t, term=0, False
            state=np.copy(state0)
            while (t<T_max) and (not term):
                action=pol[state]
                nexts, reward, term = env.step(state,action)
                state = nexts
                somme+=gamma**(t-1)*reward
                t+=1
        V.append(somme/n)
    return np.asarray(V)


pi_0=pi_0_func(N_pi)
liste_N= 30*np.arange(75)+20

Err=[np.sum(np.dot(pi_0,MC_value_func(N,pi_0,pol)-V_pi)) for N in liste_N]
        

plt.figure(4)
plt.clf()
plt.plot(liste_N, Err)
plt.title('Error evolution in function of N')


## Q5:
v_pi=[0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.82369294, 0.92820033, 0.00000000, 0.77818504, 0.82369294,
         0.87691855, 0.82847001]
         
         
def alpha(x,a,N): # learning rate choice
    return 1/N[x,a]
    

def Q_learning(Q,N,Reward,x0,T_max,eps):
    t, term=0, False
    while (t<T_max) and (not term):
        list_action=env.state_actions[x0] # possible actions in x0
        e=np.random.rand(1) # to select the action 
        if e<eps:
            a=np.random.choice(list_action)
        else :
            q=Q[x0,list_action]
            a=list_action[np.argmax(q)]
        N[x0,a]+=1
        x1, reward, term = env.step(x0,a)
        alph=alpha(x0,a,N)
        Q[x0,a]=(1-alph)*Q[x0,a]+alph*(reward+gamma*np.max(Q[x1,:])) 
        x0=x1
        Reward+=reward
        t+=1
    return Q,N,Reward


def greedy_pol(Q):
    a=np.zeros(n_states)
    for x0 in range(n_states):
        list_act=env.state_actions[x0]
        q=Q[x0,list_act]
        a[x0]=list_act[np.argmax(q)]
    return a.astype(int)


Err=[] # list of ||v-v_pi||
list_Reward=[] # list of Reward
Q=np.zeros((n_states,n_states))
N=np.zeros((n_states,n_states))
Reward=0

nb_epochs, T_max, eps = 150, 10**3, 1
list_epochs=2*np.arange(nb_epochs//2) # epochs where we calculate the value_function

for e in range(nb_epochs):
    eps *= 0.99 # we decrease the learning rate at each epoch
    x0 = env.reset()
    Q,N,Reward = Q_learning(Q,N,Reward,x0,T_max,eps)
    greed_pol = greedy_pol(Q)
    if e in list_epochs:
        list_Reward.append(Reward)
        Err.append(np.max(np.abs(MC_value_func(10**3,pi_0,greed_pol)-v_pi)))


plt.figure(5)
plt.clf()
plt.subplot(1,2,1)
plt.scatter(list_epochs, Err)
plt.title('Error evolution in function of the epoch') 
plt.subplot(1,2,2)
plt.scatter(list_epochs, list_Reward)
plt.title('Reward over the epoch')


'''
# To determine the best learning rate:
def f(lr):
    eps=1
    Q=np.zeros((n_states,n_states))
    N=np.zeros((n_states,n_states))
    Reward=0
    for e in range(nb_epochs):
        eps*=lr # we decrease the learning rate at each epoch
        x0=env.reset()
        Q,N,Reward=Q_learning(Q,N,Reward,x0,T_max,eps)
    return -Reward

from scipy.optimize import fmin
lr_best = fmin(f,0.9)
'''

## Q6: changement of the initial distribution mu0

def reset_bis(): # we define an other initial distributionu = 0.9
    a = np.random.rand(n_states)*np.arange(1,n_states+1)
    x_0 = np.argmax(np.sin(a))
    if np.random.rand()<.1:
        x0=np.random.randint(n_states)
    return x_0
    

Err=[] # list of ||v-v_pi||
list_Reward=[] # list of Reward
Q=np.zeros((n_states,n_states))
N=np.zeros((n_states,n_states))
Reward=0

nb_epochs, T_max, eps = 150, 10**3, 0.1 # constant learning rate
list_epochs=2*np.arange(nb_epochs//2) # epochs where we calculate the value_function

for e in range(nb_epochs):
    x0 = reset_bis() # we change the initialization
    Q,N,Reward = Q_learning(Q,N,Reward,x0,T_max,eps)
    greed_pol = greedy_pol(Q)
    if e in list_epochs:
        list_Reward.append(Reward)
        Err.append(np.max(np.abs(MC_value_func(10**3,pi_0,greed_pol)-v_pi)))

plt.figure(6)
plt.clf()
plt.scatter(list_epochs, Err)
plt.title('Error evolution in function of the epoch')

