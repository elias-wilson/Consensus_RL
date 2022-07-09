import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from cartpole import environment as CartPole
from cartpole_render import multi_cartpole
from Consensus_Controller import CRCC
import matplotlib.pyplot as plt
from ddpg import DDPG
from utils.noise import *
device = torch.device('cuda:0')
dtype = torch.float64

'''
Main file to for training and testing.

Package dependencies:
    Numpy
    PyTorch
    Matplotlib
    Time

File dependencies:
    cartpole.py
    cartpole_render.py (uses Open AI Gym)
    Consensus_Controller.py
    ddpg.py (uses utils folder)
'''

###############################################################################
###############################################################################
## Inputs
dt = 0.01
t = torch.arange(0,0.5,dt)

BATCHES = 100000 # Numnber of parallel simulations
EPOCHS = 10000 # Number of training steps
TRAIN_ITER = 3 # Number of traning loops during each training step
TRAIN_SIZE = int(BATCHES*(len(t)-1)/TRAIN_ITER) # Number of transitions to use per training loop
SAVE_ITER = 10  # Save every number of training iterations

H = 2 # normal agents
F = 0 # Byzantine agents
N = H + F # nomral agents
M = np.arange(F,N)

G = torch.ones((N,N),device=device,dtype=torch.long) - \
    torch.eye(N,device=device,dtype=torch.long)

measurement_matrix = torch.tensor([[1,0,0,0],[0,0,1,0]],device=device,dtype=dtype)
measurement_noise = torch.zeros((N,2),device=device,dtype=dtype)

action_space = torch.tensor([[-5,-5],[5,5]],device=device,dtype=dtype)
state_space = torch.tensor([[-10,-20,-np.pi,-2*np.pi],
                                [10,20,np.pi,2*np.pi]],device=device,dtype=dtype)

z0 = torch.randn((N,2),device=device,dtype=dtype) # Initial consensus estimate

render = True # Render during the testing phase or not

# Carpole
Cart_Params = {'L': 1, 'm': 1, 'M': 0.5, 'g': 9.81, 'dt' : dt,
                'nu': measurement_noise, 'C': measurement_matrix,
                'action_space': action_space}

# Consensus Controller
Cont_Params = {'H': H, 'F': F, 'N': N, 'lr': 0.1, 'G': G,
    'max': torch.tensor([1,np.pi],device=device,dtype=dtype),
    'min': torch.tensor([-1,-np.pi],device=device,dtype=dtype)}

checkpoint_dir = r'D:\RC_agents' # Where to save checkpoints

###############################################################################
###############################################################################
## Functions

def norm_states(x):
    return (x - state_space[0].view(1,1,-1))/(state_space[1] - \
                                                state_space[0]).view(1,1,-1)

def actor_input(x,j):
    x = norm_states(x)
    col_ids = G[j].clone()

    col_ids[j] += 1
    sum_M = int(torch.sum(col_ids))
    col_ids = col_ids*torch.arange(len(col_ids),device=device,dtype=torch.long)
    B = len(x)
    batch_ids = torch.arange(B,device=device,dtype=torch.long).view(B,1)\
        *torch.ones((B,sum_M),device=device,dtype=torch.long)
    return x[batch_ids,col_ids].reshape(B,-1)

###############################################################################
###############################################################################
## Main setup
env = CartPole(Cart_Params,device,dtype)
con = CRCC(Cont_Params,device,dtype)
# viewer = multi_cartpole(N)
BN = BATCHES*N

# Every cartpole agent has its own DDPG agent stored in 'agents'
agents = []
j = 0
for _ in range(N):
    num_inputs = int(4*(torch.sum(G[j]).item() + 1))
    agent_dir = r'\agent' + str(j)
    agents.append(DDPG(1,1e-3,[128,128,128],num_inputs,action_space,device,dtype, j,
        checkpoint_dir=checkpoint_dir+agent_dir))
    j += 1

# actor_noise = Gauss_Noise(2,0.05)
actor_noise = OUNoise((BATCHES,2), dt=dt, mu=0, theta=0.15, sigma=0.5)

start_epoch = 0

###############################################################################
###############################################################################
## Main train


# Load the checkpoint data (comment out to not load)
# j = 0
# for agent in agents:
    # Can specify path to specific file or leave as None to get most recent
    # checkpoint_path = r'D:\RC_agents\agent' + str(j) + '\ep_13250.tar'
    # start_epoch = agent.load_checkpoint(checkpoint_path=None)
    # j += 1


# Main loop
# Initiate some values
start = time.time()
r_avg = torch.zeros(N,device=device,dtype=dtype)
critic_loss_avg = 0
actor_loss_avg = 0
for i in range(start_epoch,start_epoch+EPOCHS):
    print(i,time.time()-start,torch.mean(r_avg).item(),critic_loss_avg,actor_loss_avg)

    start = time.time()
    ## Simulate system and generate data
    x = torch.zeros((BATCHES,len(t),N,4),device=device,dtype=dtype)
    x[:,0] = 2*torch.randn((BATCHES,N,4),device=device,dtype=dtype)
    z = con.projection(env.measurement(x[:,0].reshape(BN,4))).view(BATCHES,N,2)
    u = torch.zeros((BATCHES,len(t)-1,N,2),device=device,dtype=dtype)
    r = torch.zeros((BATCHES,len(t)-1,N),device=device,dtype=dtype)
    for k in range(len(t)-1):
        z = con.step(z).clone() # Update the consensus estimate

        # Here is where some consenus values can be messed with to make the Byzantine agents
        # z[:,0,0] += 100*0.5*torch.sin(0.5*np.pi*t[k])
        # z[:,0,1] += 100*0.1*torch.sin(2*np.pi*t[k])
        # z[:,1,0] += 100*torch.sin(1.5*np.pi*t[k] + 0.8)
        # z[:,1,1] += 100*2*torch.sin(0.1*np.pi*t[k] + 0.2)


        # Calculate actions for each agent by looping through 'agents'
        # Not very efficient. Could be improved
        j = 0
        for agent in agents:

            # These two lines are some simple linear controllers
            # u[:,k,j] = -15*(env.measurement(x[:,k,j]) - z[:,j]) - 5*x[:,k,j,[1,3]]
            # u[:,k,j] = torch.sigmoid(-15*env.measurement(x[:,k,j]) - 5*x[:,k,j,[1,3]])

            # These lines are using the DDPG agents
            x_in = actor_input(x[:,k],j) # Formats the input data
            u[:,k,j] = agent.calc_action(x_in,action_noise=None)
            j += 1

        # Calculates next state
        # The inputs are reshaped to matrices and the next state is coputed then reshpaped back
        # At each time step there are BATCHES*N number of cartpoles to simulate
        xp, rp = env.step(x[:,k].reshape(BN,4),u[:,k].reshape(BN,2),z.reshape(BN,2))
        x[:,k+1] = xp.view(BATCHES,N,4)
        r[:,k] = rp.view(BATCHES,N)

    actor_noise.reset()

    # Data handling
    r_avg = dt*torch.sum(r,[0,1])/BATCHES
    xp = x[:,1:].reshape(BATCHES*(len(t)-1),N,4)
    xk = x[:,:-1].reshape(BATCHES*(len(t)-1),N,4)
    u = u.reshape(BATCHES*(len(t)-1),N,2)
    r = r.reshape(BATCHES*(len(t)-1),N)

    # Training Loop
    for _ in range(TRAIN_ITER):
        train_ids = np.random.choice(BATCHES*(len(t)-1),size=TRAIN_SIZE)
        j = 0
        critic_loss_total = 0
        actor_loss_total = 0
        # Loop and train each agent in 'agents'
        # Could be more efficient
        for agent in agents:
            batch = {'state': actor_input(xk[train_ids],j), 'action': u[train_ids,j],\
            'reward': r[train_ids,j], 'next_state': actor_input(xp[train_ids],j),\
            'done' : torch.zeros((TRAIN_SIZE),device=device,dtype=dtype)}
            critic_loss, actor_loss = agent.update_params(batch)
            critic_loss_total += critic_loss
            actor_loss_total += actor_loss
            j += 1
        critic_loss_avg = critic_loss_total/j
        actor_loss_avg = actor_loss_total/j

    # print('training: '  + str(time.time()-start2))
    if ((i % SAVE_ITER == 0) or (i == start_epoch+EPOCHS-1)) and (i > start_epoch):
        print('saving')
        for agent in agents:
            agent.save_checkpoint(i)

###############################################################################
###############################################################################
## Main test

# j = 0
# for agent in agents:
#     checkpoint_path = r'D:\RC_agents\agent' + str(j) + '\ep_13250.tar'
#     start_epoch = agent.load_checkpoint(checkpoint_path=None)
#     j += 1

print(start_epoch)
## Simulate system and generate data
x = torch.zeros((BATCHES,len(t),N,4),device=device,dtype=dtype)
x[:,0] = 2*torch.randn((BATCHES,N,4),device=device,dtype=dtype)
z = con.projection(env.measurement(x[:,0].reshape(BN,4))).view(BATCHES,N,2)
u = torch.zeros((BATCHES,len(t)-1,N,2),device=device,dtype=dtype)
r = torch.zeros((BATCHES,len(t)-1,N),device=device,dtype=dtype)
zplot = np.zeros((BATCHES,len(t),N,2))
zplot[:,0] = z.cpu().numpy()
for k in range(len(t)-1):
    z = con.step(z)
    zplot[:,k+1] = z.cpu().numpy()

    # z[:,0,0] += 100*0.5*torch.sin(0.5*np.pi*t[k])
    # z[:,0,1] += 100*0.1*torch.sin(2*np.pi*t[k])
    # z[:,1,0] += 100*torch.sin(1.5*np.pi*t[k] + 0.8)
    # z[:,1,1] += 100*2*torch.sin(0.1*np.pi*t[k] + 0.2)

    j = 0
    for agent in agents:

        x_in = actor_input(x[:,k],j)
        u[:,k,j] = agent.calc_action(x_in,action_noise=actor_noise)

        # u[:,k,j] = 2*torch.sigmoid(-2.5*(env.measurement(x[:,k,j]) - z[:,j]) - 0*x[:,k,j,[1,3]])-1
        j += 1

    xp, rp = env.step(x[:,k].reshape(BN,4),u[:,k].reshape(BN,2),z.reshape(BN,2))
    x[:,k+1] = xp.view(BATCHES,N,4)
    r[:,k] = rp.view(BATCHES,N)

colors = ['r','b']
for _ in range(BATCHES):
    plot_id = np.random.choice(BATCHES)
    C = measurement_matrix.view(1,-1,4)
    plt.figure()
    plt.subplot(2,1,1)
    for i in range(N):
        y = torch.sum(C*x[plot_id,:,i].view(len(t),1,4),2).cpu()
        plt.plot(t,y,colors[i])
        plt.plot(t,zplot[plot_id,:,i],'--')
    plt.subplot(2,1,2)
    for i in range(N):
        plt.plot(t[:-1],u[plot_id,:,i].cpu(),colors[i])
    plt.show()
