# Overview

This document explain the structure of this codebase and hyperparameters of experiments. 


## File organization

### Step 1. 
Please refer to the instruction of dopamine structure in [here](https://github.com/KaixiangLin/dopamine/blob/master/docs/README.md#file-organization)

### Step 2. 
We add variants of RPG agents in [this folder](dopamine/dopamine/agents) and we explain each agent as follows:


|  Folder | Exploration  |  Supervision | 
|---|---|---|
| rpg  | epsilon-greedy  |  RPG (Hinge loss) |
| lpg  | epsilon-greedy|  LPG (Cross-Entropy) |
| epg  | EPG   | LPG (Cross-Entropy) |
|repg  | EPG   |  RPG (Hinge loss) |
|implicit_quantilerpg| implicit_quantile  |  RPG (Hinge loss) |


* EPG: EPG is the stochastic listwise policy gradient 
with off-policy supervised learning, which is the vanilla policy gradient trained 
with off-policy supervised learning. The exploration and supervision agent is parameterized 
by the same neural network. The supervision agent minimizes the cross-entropy loss 
over the near-optimal trajectories collected in an online fashion.

* LPG: LPG is the deterministic listwise policy gradient with off-policy supervised learning. 
We choose an action greedily based on the value of logits during the evaluation, and it stochastically 
explores the environment as EPG.

* RPG: RPG explores the environment using a separate agent: epsilon-greedy, EPG in Pong and 
Implicit Quantile in other games. Then rpg conducts supervised
learning by minimizing the hinge loss. 

In this codebase, the folder [rpg](dopamine/dopamine/agents/rpg) 
contain the code of RPG with epsilon-greedy exploration, and similarly [repg](dopamine/dopamine/agents/repg) for EPG exploration, 
[implicit_quantilerpg](dopamine/dopamine/agents/implicit_quantilerpg)
 for implicit quantile network exploration. 

The agents with relatively simple exploration strategy (rpg, lpg, epg, repg) perform well on Pong,
comparing to the state-of-the-arts, since there are higher chance to hit the good trajectories with in Pong. 
For more complicated games, we adopt implicit quantile network as the exploration agent. 

## Hyperparameters
The hyperparameters of networks, optimizers, etc., are same as the [baselines](https://github.com/KaixiangLin/dopamine/tree/master/baselines) in dopamine. 
The trajectory reward threshold c (see Def 5 in the paper (TODO)) for each game is given as follows:

| game  | c  |
|---|---|
|  Boxing | 100  |
|  Breakout | 400  |
|  Bowling | 80  |
|  BankHeist | 1100  |
|  DoubleDunk | 18  |
|  Pitfall | 0  |
|  Pong |  1 |
|  Robotank| 65  |





