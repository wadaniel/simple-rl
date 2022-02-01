# simple-rl
This repo contains a Python implementation of [VRACER with Ref-ER](https://www.cse-lab.ethz.ch/wp-content/papercite-data/pdf/novati2019a.pdf) and a couple of examples.
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
Tenesorflow may access the GPU and slow down execution, avoid this with
```
export CUDA_VISIBLE_DEVICES=''
```
