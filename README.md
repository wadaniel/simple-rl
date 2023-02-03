# simple-rl
This repo contains a Python implementation of [VRACER with Ref-ER](https://www.cse-lab.ethz.ch/wp-content/papercite-data/pdf/novati2019a.pdf) and a couple of examples.
VRACER Configuration parameters can be found in the constructor of the Vracer class.


## Requirements
Install python packages with
```
pip install -r requirements.txt
```

## Content 
```
simple-rl
│   README.md
│   requirements.txt
│   simpleagent.py
│   replaymemory.py
│   vracer.py
|
└───examples
    │
    └───cartpole
    │   │   README.md
    │   │   cartpole.py
    │   │   environment_simple.py
    │   │   environment_vracer.py
    │
    └───gym
        │   README.md
        │   requirements.txt
        │   environment_vracer.py

```

## GPU usage
Tenesorflow may access the GPU and slow down execution, avoid this with
```
export CUDA_VISIBLE_DEVICES=''
```
