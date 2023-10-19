# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 13:55:05 2023

@author: Xingyu
"""

import tensorflow as tf
import numpy as np
from   time import time as now
import scipy.optimize
import scipy.io as sio
import random
DTYPE='float64'
tf.keras.backend.set_floatx(DTYPE)
############################################################################################################
def sph2car(sph_coor):
    x = sph_coor[0]*np.sin(sph_coor[1])*np.cos(sph_coor[2])
    y = sph_coor[0]*np.sin(sph_coor[1])*np.sin(sph_coor[2])
    z = sph_coor[0]*np.cos(sph_coor[1])
    return [x,y,z]
############################################################################################################



########################################################################################################### 
def init_model(num_input=3, num_hidden_layers=4, neurons_per_layer=256):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(num_input))
    ################################################################
    # Append hidden layers
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(neurons_per_layer,
            activation=tf.keras.activations.get('tanh'),
            kernel_initializer='glorot_normal'))
    ################################################################
    # Output is one-dimensional
    model.add(tf.keras.layers.Dense(1))
    ################################################################
    return model
########################################################################################################### 
def get_pde(model,pde_input,wave_num1):
    with tf.GradientTape(persistent=True) as tape:
        x1, x2, x3 = pde_input[:,0:1], pde_input[:,1:2], pde_input[:,2:3]
        tape.watch(x1)
        tape.watch(x2)
        tape.watch(x3)
        pde_pred  = model(tf.stack([x1[:,0],x2[:,0],x3[:,0]],axis=1))
        x1_d1 = tape.gradient(pde_pred,x1)
        x2_d1 = tape.gradient(pde_pred,x2)
        x3_d1 = tape.gradient(pde_pred,x3)
    x1_d2 = tape.gradient(x1_d1,x1)
    x2_d2 = tape.gradient(x2_d1,x2)
    x3_d2 = tape.gradient(x3_d1,x3)
    del tape
    pde_eqn = ( x1_d2 + x2_d2 + x3_d2 )*wave_num1 + pde_pred   ## becareful here, normalization. 
    loss_pde = tf.reduce_mean(tf.square(pde_eqn))
    return loss_pde
########################################################################################################### 
def get_rigid(model,pde_rigid):
    with tf.GradientTape(persistent=True) as tape:
        x1, x2, x3 = pde_rigid[:,0:1], pde_rigid[:,1:2], pde_rigid[:,2:3]
        tape.watch(x1)
        tape.watch(x2)
        tape.watch(x3)
        pde_pred  = model(tf.stack([x1[:,0],x2[:,0],x3[:,0]],axis=1))
        x1_d1 = tape.gradient(pde_pred,x1)
        x2_d1 = tape.gradient(pde_pred,x2)
        x3_d1 = tape.gradient(pde_pred,x3)
    del tape
    pde_eqn = tf.multiply(x1_d1, x1)+ tf.multiply(x2_d1, x2)+ tf.multiply(x3_d1, x3)  ## becareful here, normalization. 
    loss_rigid = tf.reduce_mean(tf.square(pde_eqn))
    return loss_rigid
#############################################################################################################
def get_bc(model,bcc_input,bcc_target):
    bcc_pred  = model(bcc_input)
    loss_bc   = tf.reduce_mean(tf.square(bcc_pred-bcc_target)) 
    return loss_bc   
############################################################################################################# 
def get_grad(model,bcc_input,bcc_target,pde_input,pde_rigid,wave_num1):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        loss_bc   = get_bc(model,bcc_input,bcc_target)                             # boundary condition loss   
        loss_pde  = get_pde(model,pde_input,wave_num1)                            # pde loss at known 
        loss_rigid= get_rigid(model,pde_rigid)
        loss      = loss_bc + loss_pde + loss_rigid
    g = tape.gradient(loss,model.trainable_variables)
    del tape
    return loss_bc, loss_pde,loss_rigid, g
##############################################################################################################
@tf.function
def model_fit(model,bcc_input,bcc_target,pde_input,pde_rigid,wave_num1):
    loss_bc,loss_pde,loss_rigid,grad=get_grad(model,bcc_input,bcc_target,pde_input,pde_rigid,wave_num1)
    optim.apply_gradients(zip(grad,model.trainable_variables))
    return loss_bc, loss_pde,loss_rigid