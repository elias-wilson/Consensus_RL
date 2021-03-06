# Consensus_RL

## CAN NOT FIND A SOLUTION AS IS

## Description
This implents a consensus controller on a multi-agent cartpole system. A consensus controller finds a consensus value estimate for each agent, and a DDPG controller
is tasked with learning to control the system to those consensus values. The consensus controller estmates are not an input to the DPPG netowrks, so the DDPG algorithm
must learn to internally reproduce the the consensus controller outputs then drive the system to them. Hence, the true consensus controller outputs are not known when
just using the DDPG agent.

I had simplified the problem considerably to try an get the DDPG agents to find some stable solution. The files here have been returned to their normal form, but I had
turned the problem into a simple regulation one. Even then I was having significant issue getting the DDPG agent to learn. The only success I had was when I reduced
the complexity even further by making the system a single dimension integrator system. With the cartpole implementation, the critic network would reach some local 
minimum but never really converge to a solid solution, and the policy network would always go unstable.

I believe the issue is from poor data production or tuning. I am pretty sure the mechanics of the code are working correctly, but I could have missed something (always likely). There are some algorithms within the code that could be imporved, but I could not think of a great way of doing it. The simulation and consus controller are implemented in an element-wise fashtion, so they will execute rather quickly with the parallelizations built into PyTorch/Python (or however that works). The DDPG agents are stored in a list and are evaluted serially when finding actions or training (i.e., the code loops over the N number of agents to find actions or train. Within each of those N number of loops, BATCHES number of simulations are being considered element-wise.). This slows down the code somewhat, but I could not think of a way to abuse built in PyTorch functions that would work how I wanted. 

## Environmnent details:
action dimension: 2  
state dimension: 4  

## Dependencies:
PyTorch  
Numpy  
Time  
Matplotlib  
Open AI Gym  

## Files:
cartpole_main.py - the main file that is used to train and test the agents  
cartpole.py - constains the physics model of the cartpole with an NLDI application that makes the dynamics xddot = u_0 and thetaddot = u_1  
cartpole_render.py - uses Open AI Gym's rendering tool to render the multi-cartpole system  
Consensus_Controller.py - creates the consensus controller object  
ddpg.py - DDPG implementation that uses the utils folder  
