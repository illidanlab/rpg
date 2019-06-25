# Ranking Policy Gradient
Ranking Policy Gradient (RPG) is a sample-efficient off-policy policy gradient method
that learns optimal ranking of actions to maximize the return.
RPG has the following practical advantages:
- It is currently the most sample-efficient model-free algorithm for learning deterministic policies.
- It is effortless to incorporate any exploration algorithm to improve the sample-efficiency of RPG further.
- It is possible to learn a single RPG agent (parameterized by one neural network) that adapts to dynamic action space.  

This codebase contains the implementation of RPG using the
[dopamine](https://github.com/google/dopamine) framework. 


## Instructions


### Install via source
#### Step 1. 
Follow the install [instruction](https://github.com/KaixiangLin/dopamine/blob/master/README.md#install-via-source) of 
dopamine framework for [Ubuntu](https://github.com/KaixiangLin/dopamine/blob/master/README.md#ubuntu) 
or [Max OS X](https://github.com/KaixiangLin/dopamine/blob/master/README.md#mac-os-x). 

#### Step 2. 
Download the RPG source, i.e.

```
git clone git@github.com:illidanlab/rpg.git
```


## Running the tests

```
cd ./rpg/dopamine 
python -um dopamine.atari.train \
  --agent_name=rpg \
  --base_dir=/tmp/dopamine \
  --random_seed 1 \
  --game_name=Pong \
  --gin_files='dopamine/agents/rpg/configs/rpg.gin'
```

## Reproduce 
To reproduce the results in the paper, please refer to the instruction in [here](code.md). 

### Reference

If you use this RPG implementation in your work, please consider citing the following papers:
```
@article{lin2019ranking,
  title={Ranking Policy Gradient},
  author={Lin, Kaixiang and Zhou, Jiayu},
  journal={arXiv preprint arXiv:1906.09674},
  year={2019}
}
```

