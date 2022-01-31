#!/usr/bin/env python3

##  This code was inspired by the OpenAI Gym CartPole v0 environment

##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.

import math
import numpy as np, sys
from scipy.integrate import ode

class CartPole:
  def __init__(self):
    self.actionSpace = 1
    self.stateSpace = 4
    self.dt = 0.02
    self.step=0
    self.u = np.asarray([0, 0, 0, 0])     
    self.F=0
    self.t=0
    self.x_threshold = 2.4
    self.th_threshold = math.pi / 15
    self.ODE = ode(self.system).set_integrator('dopri5')

  def reset(self, seed):
    np.random.seed(seed)
    self.u = np.random.uniform(-0.05, 0.05, 4)
    self.step = 0
    self.F = 0
    self.t = 0

  def isFailed(self):
    return (abs(self.u[0])>self.x_threshold or abs(self.u[2])>self.th_threshold)

  def isOver(self):
    return self.isFailed()

  @staticmethod
  def system(t, y, act): #dynamics function
    mp, mc, l, g = 0.1, 1, 0.5, 9.81 # mass pole, mass cart, length pole, gravity
    x, v, th, w = y[0], y[1], y[2], y[3]
    costh, sinth = np.cos(th), np.sin(th)
    
    totMass = mp + mc
    tmp = (act + l * w**2 * sinth)/totMass
    wdot = (g * sinth - costh * tmp) / (l * (4.0 / 3.0 - mp * costh ** 2 / totMass))
    vdot = tmp - l * wdot * costh / totMass
    return [v, vdot, w, wdot]

  def advance(self, action):
    self.F = action[0]
    if (self.F > 10.0):
      self.F = 10.0
    elif self.F < -10.0:
      self.F = -10.0

    self.ODE.set_initial_value(self.u, self.t).set_f_params(self.F)
    self.u = self.ODE.integrate(self.t + self.dt)
    self.t = self.t + self.dt
    self.step = self.step + 1
    if self.isOver(): 
      return 1
    else: 
      return 0

  def getState(self):
    state = np.zeros(4)
    state[0] = self.u[0] # Cart Position
    state[1] = self.u[1] # Cart Velocity
    state[2] = self.u[2] # Pole Angle
    state[3] = self.u[3] # Pole Angular Velocity
    return state

  def getReward(self):
    return 1.0 - 1.0*self.isFailed();
