# simple-rl
Here is an implementation of VRACER with Ref-ER in python and a couple of examples.
VRACER Configuration parameters can be found in the ctor of the Vracer class.


## Requirements
Install python packages with
```
pip install -r requirements.txt
```

## Content 
```
simple-rl
│   README.md
│   REquirements.txt
│   replaymemory.py
│   vracer.py
|
└───examples
    │
    └───cartpole
    │   │   cartpole.py
    │   │   environment.py
    │
    └───gym
        │   environment.py

```

## GPU usage
Tnesorflow may access the GPU and slow down execution, avoid this with
```
export CUDA_VISIBLE_DEVICES=''
```
