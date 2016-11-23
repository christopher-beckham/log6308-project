import create_data
import theano
from theano import tensor as T
import numpy as np
import scipy

import pdb

import lasagne
from lasagne.layers import *
from lasagne.objectives import *
from lasagne.utils import floatX

X_train, X_valid, X_test = create_data.load_movielens10m_matrix_new()
X_train = X_train[0:1000,].todense()
X_train_rowsum = np.sum(X_train, axis=1)
X_train = X_train - X_train_rowsum

#X_train = np.random.normal(0,1, size=(1000,50)).astype("float32")

def net():
    l_in = InputLayer((None, X_train.shape[1]))
    l_dense = DenseLayer(l_in, num_units=500)
    l_dense2 = DenseLayer(l_dense, num_units=500)
    l_out = InverseLayer(l_dense2, l_dense)
    return l_out

l_out = net()
print "# params:", count_params(l_out)
X = theano.shared(X_train)
net_out = get_output(l_out, X)
params = get_all_params(l_out) # only optimise W
print "params:",params
loss = squared_error(net_out, X).mean()
grad = T.grad(loss, params)

f_loss = theano.function([], loss)
f_grad = theano.function([], grad)

#pdb.set_trace()

def unwrap(x0):
    W1_shape = np.prod( params[0].get_value().shape )
    b1_shape = np.prod( params[1].get_value().shape )
    W2_shape = np.prod( params[2].get_value().shape )
    b2_shape = np.prod( params[3].get_value().shape )
    ctr = 0
    W1_params = x0[ctr:ctr+W1_shape].reshape( params[0].get_value().shape ); ctr += W1_shape
    #pdb.set_trace()
    b1_params = x0[ctr:ctr+b1_shape].reshape( params[1].get_value().shape ); ctr += b1_shape
    W2_params = x0[ctr:ctr+W2_shape].reshape( params[2].get_value().shape ); ctr += W2_shape
    b2_params = x0[ctr::].reshape( params[3].get_value().shape ); ctr += b2_shape
    return W1_params, b1_params, W2_params, b2_params

def eval_loss(x0):
    # x0 = flat_grad_arr
    W1_params, b1_params, W2_params, b2_params = unwrap(x0)
    params[0].set_value(W1_params)
    params[1].set_value(b1_params)
    params[2].set_value(W2_params)
    params[3].set_value(b2_params)
    return f_loss().astype("float64")

def eval_grad(x0):
    W1_params, b1_params, W2_params, b2_params = unwrap(x0)
    params[0].set_value(W1_params)
    params[1].set_value(b1_params)
    params[2].set_value(W2_params)
    params[3].set_value(b2_params)    
    grads = f_grad()
    flat_grad_arr = np.hstack((grads[0].flatten(), grads[1].flatten(), grads[2].flatten(), grads[3].flatten()))
    return flat_grad_arr.astype("float64")
    #return np.array(f_grad()).flatten().astype('float64')


for i in range(0,20):
    flat_param_arr = np.hstack((
        params[0].get_value().flatten(),
        params[1].get_value().flatten(),
        params[2].get_value().flatten(),
        params[3].get_value().flatten()
    ))
    #print flat_param_arr.shape
    scipy.optimize.fmin_l_bfgs_b(eval_loss, flat_param_arr, fprime=eval_grad, maxfun=40)
    print i, f_loss()

print params[0].get_value()
