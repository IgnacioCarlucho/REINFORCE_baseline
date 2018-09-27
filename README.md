# REINFORCE

The discrete REINFORCE algorithm for solving the frozen lake problem. The environment is provided by gym. In the current example the slipery is disabled.
The actor implements the policy, while the critic is the value function. The updates are as follows: 

G ← sum Rk   
δ ← G − vˆ(S,w)     
w ← w + alpha*δ ∇ vˆ(S,w)    
θ ← θ + beta*δ∇ln( π(A|S, θ))   

If the variable baseline is disabled, the algorithm implements the vanilla REINFORCE. There is no critic and the algorithm direclty updates the policy using G, the reward returns.
This means updates: 

G ← sum Rk    
θ ← θ + beta*δ∇ln( π(A|S, θ))     


## Requirements

- Tensorflow
- Gym
- Numpy 

## Running: 

```
python main.py
```
It runs the main algorithm. Both actor and critic (value function) have 2 hidden layers with 150 neurons. I used relu for the hidden layer and of course a softmax activation for the output layer. 


## Sources: 


- [Policy Gradient Methods, From David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf)
- [Sutton](http://incompleteideas.net/book/bookdraft2018jan1.pdf)
- [DennyBritz RL examples](https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient)