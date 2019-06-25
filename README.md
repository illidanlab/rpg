# Ranking Policy Gradient
Ranking Policy Gradient (RPG) is a sample-efficienct  policy gradient method
that learns optimal ranking of actions with respect to the  long term reward.
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
TODO(RPG): 
```

## Acknowledgments
TODO(dopamine framework, fundings). 
