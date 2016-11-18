import numpy as np
from scipy.sparse import csr_matrix
import pdb
from time import time
import cPickle as pickle
import os
# --------
import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.objectives import *
from lasagne.nonlinearities import *
from lasagne.updates import *
from lasagne.utils import *
from lasagne.regularization import *

import itertools

from lasagne import init
from lasagne import nonlinearities

class DenseLowRankLayer(Layer):
    def __init__(self, incoming, num_units, W1=init.GlorotUniform(), W2=init.GlorotUniform(), k=1,
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 num_leading_axes=1, **kwargs):
        super(DenseLowRankLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units
        self.k = k
        
        if num_leading_axes >= len(self.input_shape):
            raise ValueError(
                    "Got num_leading_axes=%d for a %d-dimensional input, "
                    "leaving no trailing axes for the dot product." %
                    (num_leading_axes, len(self.input_shape)))
        elif num_leading_axes < -len(self.input_shape):
            raise ValueError(
                    "Got num_leading_axes=%d for a %d-dimensional input, "
                    "requesting more trailing axes than there are input "
                    "dimensions." % (num_leading_axes, len(self.input_shape)))
        self.num_leading_axes = num_leading_axes

        if any(s is None for s in self.input_shape[num_leading_axes:]):
            raise ValueError(
                    "A DenseLayer requires a fixed input shape (except for "
                    "the leading axes). Got %r for num_leading_axes=%d." %
                    (self.input_shape, self.num_leading_axes))
        num_inputs = int(np.prod(self.input_shape[num_leading_axes:]))
        
        # W_(num inputs x num_units) = W1_(num_inputs x m) * W2_(m x num_units)

        self.W1 = self.add_param(W1, (num_inputs, k), name="W1")
        self.W2 = self.add_param(W2, (k, num_units), name="W2")
        
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return input_shape[:self.num_leading_axes] + (self.num_units,)

    def get_output_for(self, input, **kwargs):
        num_leading_axes = self.num_leading_axes
        if num_leading_axes < 0:
            num_leading_axes += input.ndim
        if input.ndim > num_leading_axes + 1:
            # flatten trailing axes (into (n+1)-tensor for num_leading_axes=n)
            input = input.flatten(num_leading_axes + 1)

        activation = T.dot(input, T.dot(self.W1, self.W2))
        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation)


################
# data loading #
################

def load_movielens10m_matrix_new():
    data_dir = "data/movielens10m"
    ratings = np.load("%s/movielens10m.pkl_01.npy" % data_dir)
    movies = np.load("%s/movielens10m.pkl_02.npy" % data_dir)
    users = np.load("%s/movielens10m.pkl_03.npy" % data_dir)
    # randomly shuffle
    idxs = [x for x in range(0, len(ratings))]
    np.random.shuffle(idxs)
    ratings, movies, users = ratings[idxs], movies[idxs], users[idxs]
    # determine the size of the matrix
    tmp = csr_matrix((ratings, (users, movies)))
    n_users, n_items = tmp.shape
    # split into X_train, X_valid, X_test
    X_train = csr_matrix(
        (ratings[0:int(0.8*len(ratings))], (users[0:int(0.8*len(users))], movies[0:int(0.8*len(movies))])), 
        shape=(n_users, n_items)
    )
    print "X_train = ", X_train.shape
    X_valid = csr_matrix(
        (ratings[int(0.8*len(ratings)):int(0.9*len(ratings))], (users[int(0.8*len(users)):int(0.9*len(users))], movies[int(0.8*len(movies)):int(0.9*len(movies))])), 
        shape=(n_users, n_items)
    )
    print "X_valid = ", X_valid.shape
    X_test = csr_matrix(
        (ratings[int(0.9*len(ratings))::], (users[int(0.9*len(users))::], movies[int(0.9*len(movies))::])), 
        shape=(n_users, n_items)
    )
    print "X_test = ", X_test.shape
    return X_train, X_valid, X_test

# -----------

## LEGACY
"""
def load_movielens10m_matrix():
    data_dir = "data/movielens10m"
    ratings = np.load("%s/movielens10m.pkl_01.npy" % data_dir)
    movies = np.load("%s/movielens10m.pkl_02.npy" % data_dir)
    users = np.load("%s/movielens10m.pkl_03.npy" % data_dir)
    tmp = csr_matrix((ratings, (users, movies)))
    return tmp

## LEGACY
def prep_data(transpose=False):
    # load in the data, then shuffle it randomly
    X = load_movielens10m_matrix()
    idxs = [x for x in range(0, X.shape[0])]
    X = X[idxs]
    if transpose:
        X = X.T
    # split into train set, valid set, and test set
    n = X.shape[0]
    X_train, X_valid, X_test = X[0:int(0.8*n)], X[int(0.8*n):int(0.9*n)], X[int(0.9*n)::]
    print "X_train = ", X_train.shape
    print "X_valid = ", X_valid.shape
    print "X_test = ", X_test.shape
    return X_train, X_valid, X_test
"""

def select_nonzero_rows(tmp):
    return ~np.all(np.equal(tmp, 0), axis=1)

def iterator(matr, bs=32, mean_centering=False, shuffle=False):
    mat = matr.copy() # just to be safe
    if shuffle:
        idxs = [x for x in range(0, mat.shape[0])]
        mat = mat[idxs]
    b = 0
    while True:
        if b*bs >= mat.shape[0]:
            break
        if (b+1)*bs >= mat.shape[0]:
            # sparse matrix does not support over-indexing like np arrays
            this_batch = mat[(b*bs)::].todense().astype("float32")
        else:
            this_batch = mat[b*bs:(b+1)*bs].todense().astype("float32")

        #pdb.set_trace()
        
        # check: make sure this_batch has no all-zero rows
        this_batch = this_batch[ np.asarray(select_nonzero_rows(this_batch))[:,0] ]

        this_mask = (this_batch != 0).astype("float32")

        this_row_means = []
        if mean_centering:
            for i in range(0, this_batch.shape[0]):
                row = this_batch[i]
                #pdb.set_trace()
                row_mean = np.sum(row) / ((row != 0).sum()+1)
                row = row - row_mean
                this_batch[i] = row
                this_row_means.append([row_mean])
        this_row_means = np.asarray(this_row_means).astype("float32")

        # if the minibatch is all-zeros, just continue the iterator
        if this_batch.shape[0] != 0:
            yield this_batch, this_mask, this_row_means
        b += 1

####################
# lasagne networks #
####################

def i_encoder1_tieing(args):
    return _encoder1_tieing(n_items=65133, args=args)

def i_encoder1(args):
    return _encoder1(n_items=65133, args=args)

def u_encoder1(args):
    return _encoder1(n_items=71567, args=args)

def find_code_layer(l_out):
    for layer in get_all_layers(l_out):
        if layer.name == "code":
            return layer
    return None

def ui_encoder(args):
    n_users, n_items = 71567, 65133

    l_i_in = InputLayer((None, n_items))
    l_i_dense = DenseLayer(l_i_in, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"])
    l_i_dense2 = DenseLayer(l_i_dense, num_units=args["code"], nonlinearity=args["nonlinearity"])
    l_i_dense2.name = "code"
    l_i_inv = DenseLayer(l_i_dense2, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"])
    l_i_inv2 = DenseLayer(l_i_inv, num_units=n_items, nonlinearity=linear)

    l_u_in = InputLayer((None, n_users))
    l_u_dense = DenseLayer(l_u_in, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"])
    l_u_dense2 = DenseLayer(l_u_dense, num_units=args["code"], nonlinearity=args["nonlinearity"], W=l_i_dense2.W, b=l_i_dense2.b)
    l_u_dense2.name = "code"
    l_u_inv = DenseLayer(l_u_dense2, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"])
    l_u_inv2 = DenseLayer(l_u_inv, num_units=n_users, nonlinearity=linear)
    
    return l_i_inv2, l_u_inv2

def ui_encoder_lowrank(args):
    n_users, n_items = 71567, 65133

    l_i_in = InputLayer((None, n_items))
    l_i_dense = DenseLowRankLayer(l_i_in, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"], k=args["m1"])
    l_i_dense.name = "code"
    l_i_inv = DenseLayer(l_i_dense, num_units=n_items, nonlinearity=linear)

    l_u_in = InputLayer((None, n_users))
    l_u_dense = DenseLowRankLayer(l_u_in, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"], k=args["m1"], W2=l_i_dense.W2)
    l_u_dense.name = "code"
    l_u_inv = DenseLayer(l_u_dense, num_units=n_users, nonlinearity=linear)
    
    return l_i_inv, l_u_inv

"""
def ui_encoder_lowrank2(args):
    n_users, n_items = 71567, 65133

    l_i_in = InputLayer((None, n_items))
    l_i_dense = DenseLowRankLayer(l_i_in, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"], k=args["m1"])
    l_i_dense.name = "code"
    l_i_inv = DenseLowRankLayer(l_i_dense, num_units=n_items, nonlinearity=linear, k=args["k"])

    l_u_in = InputLayer((None, n_users))
    l_u_dense = DenseLowRankLayer(l_u_in, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"], k=args["m2"], W2=l_i_dense.W2)
    l_u_dense.name = "code"
    l_u_inv = DenseLowRankLayer(l_u_dense, num_units=n_users, nonlinearity=linear, k=args["k"], W1=l_i_inv.W1)
    
    return l_i_inv, l_u_inv
"""


"""
def ui_encoder_lowrank2_manual(args):
    n_users, n_items = 71567, 65133

    l_i_in = InputLayer((None, n_items))
    l_i_dense = DenseLayer(l_i_in, num_units=args["k"], nonlinearity=linear) # bottleneck to m
    l_i_dense2 = DenseLayer(l_i_dense, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"]) # bottleneck to k
    l_i_inv = DenseLayer(l_i_dense2, num_units=args["k"], nonlinearity=linear, k=args["k"]) # bottleneck to m
    l_i_inv2 = DenseLayer(l_i_inv, num_units=n_items, nonlinearity=linear) # back to the input dimension

    

    l_u_in = InputLayer((None, n_users))
    l_u_dense = DenseLowRankLayer(l_u_in, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"], k=args["k"], W2=l_i_dense.W2)
    l_u_dense.name = "code"
    l_u_inv = DenseLowRankLayer(l_u_dense, num_units=n_users, nonlinearity=linear, k=args["k"], W1=l_i_inv.W1)
    
    return l_i_inv, l_u_inv
"""



def ui_encoder_double(args):
    n_users, n_items = 71567, 65133

    l_i_in = InputLayer((None, n_items))
    l_i_dense = DenseLayer(l_i_in, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"])
    l_i_dense2 = DenseLayer(l_i_dense, num_units=args["code"], nonlinearity=args["nonlinearity"])
    l_i_dense2.name = "code"
    l_i_inv = DenseLayer(l_i_dense2, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"])
    l_i_inv2 = DenseLayer(l_i_inv, num_units=n_items, nonlinearity=linear)

    l_u_in = InputLayer((None, n_users))
    l_u_dense = DenseLayer(l_u_in, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"])
    l_u_dense2 = DenseLayer(l_u_dense, num_units=args["code"], nonlinearity=args["nonlinearity"], W=l_i_dense2.W, b=l_i_dense2.b)
    l_u_dense2.name = "code"
    l_u_inv = DenseLayer(l_u_dense2, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"], W=l_i_inv.W, b=l_i_inv.b)
    l_u_inv2 = DenseLayer(l_u_inv, num_units=n_users, nonlinearity=linear)
    
    return l_i_inv2, l_u_inv2



def i_simple_net_lowrank2(args):
    return _simple_net_lowrank2(n_items=65133, args=args)

def u_simple_net_lowrank2(args):
    return _simple_net_lowrank2(n_items=71567, args=args)

def _simple_net_lowrank2(n_items, args):
    l_in = InputLayer((None, n_items))
    if "sigma" in args:
        print "adding gauss noise: %f" % args["sigma"]
        l_in = GaussianNoiseLayer(l_in, args["sigma"])
    l_dense = DenseLowRankLayer(l_in, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"], k=args["m1"])
    l_dense.name = "code"
    l_inv = DenseLowRankLayer(l_dense, num_units=n_items, nonlinearity=linear, k=args["m2"])
    return l_inv


def i_simple_net_lowrank(args):
    return _simple_net_lowrank(n_items=65133, args=args)

def u_simple_net_lowrank(args):
    return _simple_net_lowrank(n_items=71567, args=args)

def _simple_net_lowrank(n_items, args):
    l_in = InputLayer((None, n_items))
    if "sigma" in args:
        print "adding gauss noise: %f" % args["sigma"]
        l_in = GaussianNoiseLayer(l_in, args["sigma"])
    l_dense = DenseLowRankLayer(l_in, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"], k=args["m1"])
    l_dense.name = "code"
    l_inv = DenseLayer(l_dense, num_units=n_items, nonlinearity=linear)
    return l_inv




def i_simple_net_lowrank2_manual(args):
    return _simple_net_lowrank2_manual(n_items=65133, args=args)

def u_simple_net_lowrank2_manual(args):
    return _simple_net_lowrank2_manual(n_items=71567, args=args)

def _simple_net_lowrank2_manual(n_items, args):
    l_in = InputLayer((None, n_items))
    l_dense = DenseLayer(l_in, nonlinearity=linear, num_units=args["k"]) # bottleneck to m
    l_dense2 = DenseLayer(l_dense, nonlinearity=args["nonlinearity"], num_units=args["bottleneck"]) # bottleneck to k
    l_dense2.name = "code"
    l_inv = DenseLayer(l_dense2, num_units=args["k"], nonlinearity=linear) # bottleneck to m
    l_inv2 = DenseLayer(l_inv, num_units=n_items, nonlinearity=linear) # bottleneck to original input dim
    return l_inv2





def i_simple_net(args):
    return _simple_net(n_items=65133, args=args)

def u_simple_net(args):
    return _simple_net(n_items=71567, args=args)

# u/i agnostic implementations

def _encoder1_tieing(n_items, args):
    l_in = InputLayer((None, n_items))
    l_dense = DenseLayer(l_in, num_units=args["bottleneck"])
    l_dense2 = DenseLayer(l_dense, num_units=args["code"])
    l_inv = l_dense2
    for layer in get_all_layers(l_inv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        l_inv = InverseLayer(l_inv, layer)
    return l_inv

def _encoder1(n_items, args):
    l_in = InputLayer((None, n_items))
    l_dense = DenseLayer(l_in, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"])
    l_dense2 = DenseLayer(l_dense, num_units=args["code"], nonlinearity=args["nonlinearity"])
    l_dense2.name = "code"
    l_inv = DenseLayer(l_dense2, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"])
    l_inv2 = DenseLayer(l_inv, num_units=n_items, nonlinearity=linear)
    return l_inv2


def _simple_net(n_items, args):
    l_in = InputLayer((None, n_items))
    if "sigma" in args:
        print "adding gauss noise: %f" % args["sigma"]
        l_in = GaussianNoiseLayer(l_in, args["sigma"])
    l_dense = DenseLayer(l_in, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"])
    l_dense.name = "code"
    l_inv = DenseLayer(l_dense, num_units=n_items, nonlinearity=linear)
    return l_inv

# --

def simple_net_tieing(args):
    n_items = 65133
    l_in = InputLayer((None, n_items))
    l_dense = DenseLayer(l_in, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"])
    l_dense.name = "code"
    l_inv = InverseLayer(l_dense, l_dense)
    return l_inv


# ################
# lasagne helper #
# ################

def get_net(net_fns, args):
    print "compiling theano functions..."
    X_I = T.fmatrix('X_I')
    X_U = T.fmatrix('X_U')
    M_I = T.fmatrix('M_I')
    M_U = T.fmatrix('M_U')
    # get the item encoder and the user encoder
    # if net_fns is a tuple
    if type(net_fns) is tuple:
        l_out_i, l_out_u = net_fns[0](args), net_fns[1](args)
    else:
        l_out_i, l_out_u = net_fns(args)
    print_network(l_out_i)
    print_network(l_out_u)
    net_out_i, net_out_u = get_output(l_out_i, X_I), get_output(l_out_u, X_U)
    assert args["mode"] in ["item", "user", "item_mask", "user_mask", "both", "both_mask"]
    if "l2" not in args:
        args["l2"] = 0.0
    if args["mode"] == "item":
        loss = squared_error(net_out_i, X_I).mean() + args["l2"]*regularize_network_params(l_out_i, l2)
        loss_i, loss_u = loss, T.constant(0.0)
        params = get_all_params(l_out_i, trainable=True)
    elif args["mode"] == "user":
        loss = squared_error(net_out_u, X_U).mean() + args["l2"]*regularize_network_params(l_out_u, l2)
        loss_i, loss_u = T.constant(0.0), loss
        params = get_all_params(l_out_u, trainable=True)
    elif args["mode"] == "item_mask":
        loss = (M_I*squared_error(net_out_i, X_I)).mean() + args["l2"]*regularize_network_params(l_out_i, l2)
        loss_i, loss_u = (M_I*squared_error(net_out_i, X_I)).mean(), T.constant(0.0)        
        params = get_all_params(l_out_i, trainable=True)
    elif args["mode"] == "user_mask":
        loss = (M_U*squared_error(net_out_u, X_U)).mean() + args["l2"]*regularize_network_params(l_out_u, l2)
        loss_i, loss_u = T.constant(0.0), (M_U*squared_error(net_out_u, X_U)).mean()
        params = get_all_params(l_out_u, trainable=True)
    elif args["mode"] == "both":
        loss_i = squared_error(net_out_i, X_I).mean() + args["l2"]*regularize_network_params(l_out_i, l2)
        loss_u = squared_error(net_out_u, X_U).mean() + args["l2"]*regularize_network_params(l_out_u, l2)
        loss = loss_i + loss_u
        params_set = set()
        for elem in get_all_params(l_out_i, trainable=True):
            params_set.add(elem)
        for elem in get_all_params(l_out_u, trainable=True):
            params_set.add(elem)
        params = list(params_set)
    elif args["mode"] == "both_mask":
        loss = (M_I*squared_error(net_out_i, X_I)).mean() + (M_I*squared_error(net_out_u, X_U)).mean + \
          args["l2"]*regularize_network_params(l_out_i, l2) + args["l2"]*regularize_network_params(l_out_u, l2)
        loss_i = squared_error(net_out_i, X_I).mean()
        loss_u = squared_error(net_out_u, X_U).mean()
        params = get_all_params(l_out_i, trainable=True)
        params += get_all_params(l_out_u, trainable=True)        
        
    print "params =", params
    lr = theano.shared(floatX(args["learning_rate"]))
    if args["optimiser"] == "nesterov_momentum":
        updates = nesterov_momentum(loss, params, learning_rate=lr, momentum=0.9)
    elif args["optimiser"] == "adam":
        updates = adam(loss, params, learning_rate=lr)
    train_fn = theano.function([X_I,M_I,X_U,M_U], [loss, loss_i, loss_u], updates=updates, on_unused_input='warn')
    loss_fn = theano.function([X_I,M_I,X_U,M_U], [loss, loss_i, loss_u], on_unused_input='warn')
    out_fn_i = theano.function([X_I], net_out_i)
    out_fn_u = theano.function([X_U], net_out_u)
    code_fn_i = theano.function([X_I], get_output( find_code_layer(l_out_i), X_I) )
    code_fn_u = theano.function([X_U], get_output( find_code_layer(l_out_u), X_U) )
    
    return {
        "train_fn": train_fn,
        "loss_fn": loss_fn,
        "code_fn_i": code_fn_i,
        "code_fn_u": code_fn_u,
        "out_fn_i":out_fn_i,
        "out_fn_u":out_fn_u,
        "lr": lr,
        "l_out_i": l_out_i,
        "l_out_u": l_out_u
    }

def print_network(l_out):
    for layer in get_all_layers(l_out):
        print layer, layer.output_shape
    print "number of params: %i" % count_params(layer)

def mean_center(X_batch):
    """
    X_batch is modified in place
    returns: row mean vector
    """
    row_means = []
    for i in range(0, X_batch.shape[0]):
        row = X_batch[i]
        row_mean = np.sum(row) / ((row != 0).sum()+1)
        row = row - row_mean
        X_batch[i] = row
        row_means.append([row_mean])
    row_means = np.asarray(row_means, dtype="float32")
    return row_means
    
def test(net_cfg, X_test, mean_centering=False):
    # mean center the test set
    row_means = []
    if mean_centering:
        mean_center(X_batch)
    # get the predictions
    out_fn = net_cfg["out_fn"]
    X_test_reconstruction = out_fn(X_test)
    # add the row means back in

    pass

def restore_model(net_cfg, model_file):
    if not os.path.exists(model_file):
        print "no model found, resuming..."
        return
    with open(model_file) as g:
        dat = pickle.load(g)
        set_all_param_values(net_cfg["l_out_i"], dat[0])
        set_all_param_values(net_cfg["l_out_u"], dat[1])

def dump_code_layer(out_file, net_cfg, mode, model_file, X_full, mean_centering):
    with open(out_file, "wb") as f:
        print "restoring model..."
        restore_model(net_cfg, model_file)
        assert mode in ["item", "user"]
        if mode == "item":
            code_fn = net_cfg["code_fn_i"]
        elif mode == "user":
            code_fn = net_cfg["code_fn_u"]
        buf = []
        for X_batch, _, _ in iterator(X_full, bs=128, shuffle=False, mean_centering=mean_centering):
            codes = code_fn(X_batch)
            buf.append(codes)
        buf = np.vstack(buf)
        np.save(out_file, buf)

def dump_out_layer(out_file, net_cfg, mode, model_file, X_full, mean_centering, limit=50):
    with open(out_file, "wb") as f:
        print "restoring model..."
        restore_model(net_cfg, model_file)
        assert mode in ["item", "user"]
        if mode == "item":
            out_fn = net_cfg["out_fn_i"]
        elif mode == "user":
            out_fn = net_cfg["out_fn_u"]
        buf = []
        for X_batch, _, _ in iterator(X_full, bs=128, shuffle=False, mean_centering=mean_centering):
            codes = out_fn(X_batch)
            buf.append(codes)
            if len(buf) > limit:
                break
        buf = np.vstack(buf)
        np.save(out_file, buf)

        
def dump_users_to_disk(out_file, X_full, mean_centering):
    print "dumping X (users) to disk..."
    for X_batch, _ in iterator(X_full, bs=X_full.shape[0], shuffle=False, mean_centering=mean_centering):
        print X_batch.shape
        break
    np.save(out_file, X_batch)

def train(net_cfg, data, num_epochs, batch_size, out_dir, model_dir, schedule={}, resume=None, shuffle=False, mean_centering=False, quick_check=False, debug=False):
    print "training..."
    #pdb.set_trace()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    headers = ["epoch", "train_loss", "train_i_loss", "train_u_loss", "valid_loss", "valid_i_loss", "valid_u_loss", "valid_i_rmse", "valid_u_rmse", "learning_rate", "time"]
    l_out_i, l_out_u, train_fn, loss_fn, lr = net_cfg["l_out_i"], net_cfg["l_out_u"], net_cfg["train_fn"], net_cfg["loss_fn"], net_cfg["lr"]
    if resume != None:
        write_flag = "ab"
        print "resuming training..."
        restore_model(net_cfg, resume)
    else:
        write_flag = "wb"
    f_clean = open("%s/results.txt" % out_dir, write_flag)
    if write_flag == "wb":
        f_clean.write(",".join(headers) + "\n")
        print ",".join(headers)
    f_train_raw = open("%s/results_train_raw.txt" % out_dir, write_flag)
    f_valid_raw = open("%s/results_valid_raw.txt" % out_dir, write_flag)
    X_train, X_valid, X_test = data
    for epoch in range(0, num_epochs):
        t0 = time()
        if epoch+1 in schedule:
            print "changing learning rate to: %f\n" % schedule[epoch+1]
            lr.set_value( schedule[epoch+1] )
        train_losses, train_losses_i, train_losses_u = [], [], []
        i_iterator = iterator(X_train, bs=batch_size, shuffle=shuffle, mean_centering=mean_centering)
        u_iterator = iterator(X_train.T, bs=batch_size, shuffle=shuffle, mean_centering=mean_centering)
        for I_tp, U_tp in itertools.izip(i_iterator, u_iterator):
            X_batch_I, X_mask_I, X_batch_I_rowmeans = I_tp
            X_batch_U, X_mask_U, X_batch_U_rowmeans = U_tp
            #pdb.set_trace()
            if debug:
                pdb.set_trace()
            this_loss, this_loss_i, this_loss_u = train_fn(X_batch_I, X_mask_I, X_batch_U, X_mask_U)
            f_train_raw.write("%f\n" % this_loss); f_train_raw.flush()
            train_losses.append(this_loss)
            train_losses_i.append(this_loss_i)
            train_losses_u.append(this_loss_u)
            if quick_check:
                break
        avg_train_loss, avg_train_loss_i, avg_train_loss_u = np.mean(train_losses), np.mean(train_losses_i), np.mean(train_losses_u)
        # TODO: refactor
        valid_losses, valid_losses_i, valid_losses_u, i_rmses, u_rmses = [], [], [], [], []
        i_iterator = iterator(X_valid, bs=batch_size, shuffle=shuffle, mean_centering=mean_centering)
        u_iterator = iterator(X_valid.T, bs=batch_size, shuffle=shuffle, mean_centering=mean_centering)
        for I_tp, U_tp in itertools.izip(i_iterator, u_iterator):
            X_batch_I, X_mask_I, X_batch_I_rowmeans = I_tp
            X_batch_U, X_mask_U, X_batch_U_rowmeans = U_tp
            this_loss, this_loss_i, this_loss_u = loss_fn(X_batch_I, X_mask_I, X_batch_U, X_mask_U)
            f_valid_raw.write("%f\n" % this_loss); f_valid_raw.flush()
            valid_losses.append(this_loss)
            valid_losses_i.append(this_loss_i)
            valid_losses_u.append(this_loss_u)
            # -- do actual item rmse --
            X_batch_I = np.asarray(X_batch_I) # hack
            X_batch_I_reconstruct = net_cfg["out_fn_i"](X_batch_I)
            #pdb.set_trace()
            i_rmse = np.sqrt( np.sum( np.multiply( (X_batch_I-X_batch_I_reconstruct)**2,X_mask_I) ) / np.sum(X_mask_I) )
            i_rmses.append(i_rmse)
            # -- do actual user rmse --
            X_batch_U = np.asarray(X_batch_U) # hack
            X_batch_U_reconstruct = net_cfg["out_fn_u"](X_batch_U)
            u_rmse = np.sqrt( np.sum( np.multiply( (X_batch_U-X_batch_U_reconstruct)**2,X_mask_U) ) / np.sum(X_mask_U) )
            u_rmses.append(u_rmse)
            if quick_check:
                break
            
        avg_valid_loss, avg_valid_loss_i, avg_valid_loss_u, avg_i_rmse, avg_u_rmse = \
          np.mean(valid_losses), np.mean(valid_losses_i), np.mean(valid_losses_u), np.mean(i_rmses), np.mean(u_rmses)
        out_str = "%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f" % \
          (epoch+1, avg_train_loss, avg_train_loss_i, avg_train_loss_u, avg_valid_loss, avg_valid_loss_i, avg_valid_loss_u, avg_i_rmse, avg_u_rmse, lr.get_value(), time()-t0)
        f_clean.write("%s\n" % out_str)
        f_clean.flush()
        print out_str
        with open("%s/%i.model" % (model_dir, epoch+1), "wb") as g:
            pickle.dump( [ get_all_param_values(l_out_i), get_all_param_values(l_out_u) ], g, pickle.HIGHEST_PROTOCOL)
            #np.savez(g, get_all_param_values(l_out_i), get_all_param_values(l_out_u) )

if __name__ == '__main__':

    #np.random.seed(0)

    #######################
    # item-encoder models #
    # #####################
    
    if "C200_C200_ADAM" in os.environ:
        bottleneck, code = 200, 200
        net1_cfg = get_net(i_encoder1, {"bottleneck":bottleneck, "code":code, "learning_rate":0.01, "nonlinearity":sigmoid, "optimiser":"adam"})
        dirname = "i_encoder_c%i_c%i_lr0.01_adam" % (bottleneck, code)
        train(net1_cfg,
            data=prep_data(),
            num_epochs=100,
            batch_size=128,
            shuffle=True,
            mean_centering=True,
            out_dir="output/%s" % dirname,
            model_dir="/state/partition3/cbeckham/%s" % dirname,
            schedule={50: 0.001}
        )

    
    if "C500_C500" in os.environ:
        bottleneck, code = 500, 500
        net1_cfg = get_net(i_encoder1, {"bottleneck":bottleneck, "code":code, "learning_rate":0.01, "nonlinearity":sigmoid })
        dirname = "i_encoder_c%i_c%i_lr0.01" % (bottleneck, code)
        train(net1_cfg,
            data=prep_data(),
            num_epochs=100,
            batch_size=128,
            shuffle=True,
            mean_centering=True,
            out_dir="output/%s" % dirname,
            model_dir="/state/partition3/cbeckham/%s" % dirname,
            schedule={50: 0.001},
            resume="/usagers/cbeckham/github/log6308-project/output/i_encoder_c500_c500_lr0.01/10.model"
        )

    if "MASK200" in os.environ:
        for bottleneck in [50, 100, 300, 400, 500]:
            dirname = "i_simple_net_c%i_mask" % bottleneck
            net1_cfg = get_net(i_simple_net, {"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "mask":True})
            train(net1_cfg, data=prep_data(), num_epochs=10, batch_size=128, shuffle=True, mean_centering=True,
                      model_dir="/state/partition3/cbeckham/%s" % dirname,
                      out_dir="output/%s" % dirname)

    if "SINGLE_LAYER_ADAM" in os.environ:
        for bottleneck in [50, 100, 200, 300, 400, 500]:
            dirname = "i_simple_net_c%i_adam" % bottleneck
            net1_cfg = get_net(i_simple_net, {"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"adam"})
            train(net1_cfg, data=prep_data(), num_epochs=10, batch_size=128, shuffle=True, mean_centering=True,
                      model_dir="/state/partition3/cbeckham/%s" % dirname,
                      out_dir="output/%s" % dirname)

    #######################
    # user-encoder models #
    # #####################

    if "U_SINGLE_LAYER" in os.environ:
        for bottleneck in [50, 100, 200, 300, 400, 500]:
            dirname = "u_simple_net_c%i" % bottleneck
            net1_cfg = get_net(u_simple_net, {"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum"})
            train(net1_cfg, data=prep_data(transpose=True), num_epochs=10, batch_size=128, shuffle=True, mean_centering=True,
                      model_dir="/state/partition3/cbeckham/%s" % dirname,
                      out_dir="output/%s" % dirname)


    ###################
    # dump code layer #
    ###################

    if "CODE_HYBRID_ITEM_TANH_50" in os.environ:
        bottleneck = 25
        dirname = "hybrid_item_tanh_c%i" % bottleneck
        net1_cfg = get_net((i_simple_net, u_simple_net),{"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":tanh, "optimiser":"nesterov_momentum", "mode":"item"})
        dump_code_layer("codes/%s.npy" % dirname, net1_cfg, mode="item", model_file="/state/partition4/cbeckham/hybrid_item_tanh_c25/30.model", X_full=load_movielens10m_matrix_new()[0], mean_centering=True) # train set
            

    ############
    # new data #
    ############

    if "U_NEW_SINGLE_LAYER" in os.environ:
        for bottleneck in [50]:
            dirname = "u_new_simple_net_c%i" % bottleneck
            net1_cfg = get_net(u_simple_net, {"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum"})
            train(net1_cfg, data=load_movielens10m_matrix_new(transpose=True), num_epochs=10, batch_size=128, shuffle=True, mean_centering=True,
                      model_dir="/state/partition3/cbeckham/%s" % dirname,
                      out_dir="output_new/%s" % dirname)
    


    if "HYBRID_ITEM_RELU" in os.environ:
        for bottleneck in [5000]:
            dirname = "hybrid_item_relu_c%i" % bottleneck
            net1_cfg = get_net((i_simple_net, u_simple_net),{"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":rectify, "optimiser":"nesterov_momentum", "mode":"item"})
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=15, batch_size=128, shuffle=True, mean_centering=True,model_dir="/storeSSD/cbeckham/log6308_models/%s" % dirname,out_dir="output_new2/%s" % dirname)            


    # output_new3
            
    if "HYBRID_ITEMMASK_RELU" in os.environ:
        for bottleneck in [10,50,200]:
            dirname = "hybrid_itemmask_relu_c%i_lr0.5" % bottleneck
            net1_cfg = get_net((i_simple_net, u_simple_net),{"bottleneck":bottleneck, "learning_rate":0.5, "nonlinearity":rectify, "optimiser":"nesterov_momentum", "mode":"item_mask"})
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=15, batch_size=128, shuffle=True, mean_centering=True,model_dir="/storeSSD/cbeckham/log6308_models/%s" % dirname,out_dir="output_new3/%s" % dirname)

    if "HYBRID_ITEMMASK_RELU_L2" in os.environ:
        for l2_coef in [1e-4, 1e-3, 1e-5]:
            for bottleneck in [100]:
                dirname = "hybrid_itemmask_relu_c%i_l2-%f_lr0.5" % (bottleneck, l2_coef)
                net1_cfg = get_net((i_simple_net, u_simple_net),{"bottleneck":bottleneck, "learning_rate":0.5, "nonlinearity":rectify, "optimiser":"nesterov_momentum", "mode":"item_mask", "l2":l2_coef})
                train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=15, batch_size=128, shuffle=True, mean_centering=True,model_dir="/storeSSD/cbeckham/log6308_models/%s" % dirname,out_dir="output_new3/%s" % dirname)

            

    # ----------




            
    if "HYBRID_ITEMMASK_TANH" in os.environ:
        for bottleneck in [10,50,100,200]:
            dirname = "hybrid_itemmask_tanh_c%i" % bottleneck
            net1_cfg = get_net((i_simple_net, u_simple_net),{"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":tanh, "optimiser":"nesterov_momentum", "mode":"item_mask"})
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=15, batch_size=128, shuffle=True, mean_centering=True,model_dir="/storeSSD/cbeckham/log6308_models/%s" % dirname,out_dir="output_new2/%s" % dirname)
            
            
    if "HYBRID_USERMASK_RELU" in os.environ:
        for bottleneck in [2000]:
            dirname = "hybrid_usermask_relu_c%i" % bottleneck
            net1_cfg = get_net((i_simple_net, u_simple_net),{"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":rectify, "optimiser":"nesterov_momentum", "mode":"user_mask" })
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=15, batch_size=128, shuffle=True, mean_centering=True,model_dir="/storeSSD/cbeckham/log6308_models/%s" % dirname,out_dir="output_new2/%s" % dirname)

            
    if "HYBRID_USERMASK_TANH" in os.environ:
        for bottleneck in [200, 100, 50, 10]:
            dirname = "hybrid_usermask_tanh_c%i" % bottleneck
            net1_cfg = get_net((i_simple_net, u_simple_net),{"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":tanh, "optimiser":"nesterov_momentum", "mode":"user_mask" })
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=15, batch_size=128, shuffle=True, mean_centering=True,model_dir="/storeSSD/cbeckham/log6308_models/%s" % dirname,out_dir="output_new2/%s" % dirname)


            

    if "HYBRID_USER_RELU" in os.environ:
        for bottleneck in [5,10,25,50]:
            dirname = "hybrid_user_relu_c%i" % bottleneck
            net1_cfg = get_net((i_simple_net, u_simple_net),{"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":rectify, "optimiser":"nesterov_momentum", "mode":"user"})
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=15, batch_size=128, shuffle=True, mean_centering=True,model_dir="/storeSSD/cbeckham/log6308_models/%s" % dirname,out_dir="output_new2/%s" % dirname)

            
            
    if "HYBRID_ITEM" in os.environ:
        for bottleneck in [1000]:
            dirname = "hybrid_item_c%i" % bottleneck
            net1_cfg = get_net((i_simple_net, u_simple_net),{"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"item"})
            if "CODEDUMP" not in os.environ:
                train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=15, batch_size=128, shuffle=True, mean_centering=True,model_dir="/storeSSD/cbeckham/log6308_models/%s" % dirname,out_dir="output_new2/%s" % dirname)
            else:
                dump_out_layer("codes/%s.npy" % dirname, net1_cfg, mode="item", model_file="/storeSSD/cbeckham/log6308_models/hybrid_item_c50/30.model", X_full=load_movielens10m_matrix_new()[0], mean_centering=True) # valid set

    if "VIS" in os.environ:
        net1_cfg = get_net((i_simple_net, u_simple_net),{"bottleneck":10, "learning_rate":0.1, "nonlinearity":rectify, "optimiser":"nesterov_momentum", "mode":"item_mask"})
        #dump_out_layer("codes/test3.out.npy", net1_cfg, mode="item", model_file="/storeSSD/cbeckham/log6308_models/hybrid_item_relu_c10/9.model", X_full=load_movielens10m_matrix_new()[0], mean_centering=True, limit=10)
        dump_code_layer("codes/test3.code.npy", net1_cfg, mode="item", model_file="/storeSSD/cbeckham/log6308_models/hybrid_itemmask_relu_c10/10.model", X_full=load_movielens10m_matrix_new()[0], mean_centering=True)
        

    if "HYBRID_ITEM_LOWRANK" in os.environ:
        #seed=0
        #np.random.seed(seed)
        for bottleneck in [50]:
            for m1 in [5]:
                dirname = "hybrid_item_lowrank2_m1-%i_c%i" % (m1,bottleneck)
                net1_cfg = get_net((i_simple_net_lowrank, u_simple_net_lowrank),{"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"item", "m1":m1})
                #train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=15, batch_size=128, shuffle=True, mean_centering=True,model_dir="/storeSSD/cbeckham/log6308_models/%s" % dirname,out_dir="output_new/%s" % dirname
                dump_out_layer("codes/%s.deleteme.epoch10.out.npy" % dirname, net1_cfg, mode="item", model_file="/storeSSD/cbeckham/log6308_models/%s/10.model" % dirname, X_full=load_movielens10m_matrix_new()[1], mean_centering=True) # valid set


    if "HYBRID_USER_LOWRANK" in os.environ:
        seed = 1
        np.random.seed(seed)
        for bottleneck in [50]:
            for m1 in [5,10,20,30]:
                dirname = "hybrid_user_lowrank2_m1-%i_c%i.%i" % (m1,bottleneck,seed)
                net1_cfg = get_net((i_simple_net_lowrank, u_simple_net_lowrank),{"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"user", "m1":m1})
                train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=15, batch_size=128, shuffle=True, mean_centering=True,model_dir="/storeSSD/cbeckham/log6308_models/%s" % dirname,out_dir="output_new/%s" % dirname)


                
    if "HYBRID_ITEM_LOWRANK_MANUAL" in os.environ:
        for bottleneck in [500]:
            k = 350
            dirname = "hybrid_item_lowrank_manual_k%i_c%i" % (k,bottleneck)
            net1_cfg = get_net((i_simple_net_lowrank2_manual, u_simple_net_lowrank2_manual),{"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"item", "k":k})
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=15, batch_size=128, shuffle=True, mean_centering=True,model_dir="/storeSSD/cbeckham/log6308_models/%s" % dirname,out_dir="output_new/%s" % dirname)


                
    if "HYBRID_ITEM_ADAM" in os.environ:
        for bottleneck in [25,50]:
            dirname = "hybrid_item_adam_c%i" % bottleneck
            net1_cfg = get_net((i_simple_net, u_simple_net), {"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"adam", "mode":"item" })
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=15, batch_size=128, shuffle=True, mean_centering=True, model_dir="/state/partition4/cbeckham/%s" % dirname,out_dir="output_new/%s" % dirname)
                
    if "HYBRID_ITEM_SIGMA001" in os.environ:
        sigma = 0.01
        for bottleneck in [25]:
            dirname = "hybrid_item_s-%f_c%i" % (sigma, bottleneck)
            net1_cfg = get_net((i_simple_net, u_simple_net), {"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"item", "sigma":sigma})
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=15, batch_size=128, shuffle=True, mean_centering=True,model_dir="/state/partition4/cbeckham/%s" % dirname,out_dir="output_new/%s" % dirname)

    if "HYBRID_ITEM_SIGMA005" in os.environ:
        sigma = 0.05
        for bottleneck in [25]:
            dirname = "hybrid_item_s-%f_c%i" % (sigma, bottleneck)
            net1_cfg = get_net((i_simple_net, u_simple_net), {"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"item", "sigma":sigma})
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=30, batch_size=128, shuffle=True, mean_centering=True,model_dir="/state/partition4/cbeckham/%s" % dirname,out_dir="output_new/%s" % dirname)

    if "HYBRID_ITEM_SIGMA01" in os.environ:
        sigma = 0.1
        for bottleneck in [25]:
            dirname = "hybrid_item_s-%f_c%i" % (sigma, bottleneck)
            net1_cfg = get_net((i_simple_net, u_simple_net), {"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"item", "sigma":sigma})
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=30, batch_size=128, shuffle=True, mean_centering=True,model_dir="/state/partition4/cbeckham/%s" % dirname,out_dir="output_new/%s" % dirname)

            
            
    if "HYBRID_ITEM_ADAM_L2" in os.environ:
        for l2_coef in [1e-5, 1e-4, 1e-3, 1e-2]:
            for bottleneck in [25]:
                dirname = "hybrid_item_adam_l2-%f_c%i" % (l2_coef, bottleneck)
                net1_cfg = get_net((i_simple_net, u_simple_net), {"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"adam", "mode":"item", "l2":l2_coef})
                train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=40, batch_size=128, shuffle=True, mean_centering=True,model_dir="/state/partition4/cbeckham/%s" % dirname,out_dir="output_new/%s" % dirname)

    if "HYBRID_ITEM_L2" in os.environ:
        for l2_coef in [1e-4]:
            for bottleneck in [25]:
                dirname = "hybrid_item_l2-%f_c%i" % (l2_coef, bottleneck)
                net1_cfg = get_net((i_simple_net, u_simple_net), {"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"item", "l2":l2_coef})
                train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=100, batch_size=1024, shuffle=True, mean_centering=True,model_dir="/storeSSD/cbeckham/log6308_models/%s" % dirname,out_dir="output_new/%s" % dirname, resume="/storeSSD/cbeckham/log6308_models/hybrid_item_l2-0.000100_c25/40.model.bak")

                
    if "HYBRID_ITEM_MASK" in os.environ:
        for bottleneck in [25, 100, 200]:
            dirname = "hybrid_item_mask_c%i" % bottleneck
            net1_cfg = get_net((i_simple_net, u_simple_net), {"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"item_mask"})
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=15, batch_size=128, shuffle=True, mean_centering=True,model_dir="/state/partition4/cbeckham/%s" % dirname,out_dir="output_new/%s" % dirname)
            
    if "HYBRID_USER" in os.environ:
        for bottleneck in [30]:
            dirname = "hybrid_user_c%i" % bottleneck
            net1_cfg = get_net((i_simple_net, u_simple_net), {"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"user"})
            #train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=15, batch_size=128, shuffle=True, mean_centering=True, model_dir="/storeSSD/cbeckham/log6308_models/%s" % dirname, out_dir="output_new/%s" % dirname)

    if "HYBRID_USER_MASK" in os.environ:
        for bottleneck in [200]:
            dirname = "hybrid_user_mask_c%i" % bottleneck
            net1_cfg = get_net((i_simple_net, u_simple_net),{"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"user_mask"})
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=30, batch_size=128, shuffle=True, mean_centering=True, model_dir="/state/partition4/cbeckham/%s" % dirname, out_dir="output_new/%s" % dirname)         

    if "HYBRID_BOTH" in os.environ:
        for bottleneck in [50]:
            dirname = "hybrid_both_c%i" % bottleneck
            net1_cfg = get_net((i_simple_net, u_simple_net),{"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"both"})
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=30, batch_size=128, shuffle=True, mean_centering=True,model_dir="/state/partition3/cbeckham/%s" % dirname,out_dir="output_new/%s" % dirname)

    # --------------------------
            
    if "HYBRID_ITEM_DEEPER" in os.environ:
        for bottleneck in [25, 50]:
            dirname = "hybrid_item_deeper_c%i" % bottleneck
            net1_cfg = get_net((i_encoder1, u_encoder1),{"bottleneck":bottleneck, "code":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"item"})
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=30, batch_size=128, shuffle=True, mean_centering=True,model_dir="/state/partition4/cbeckham/%s" % dirname,out_dir="output_new/%s" % dirname)

    if "HYBRID_ITEM_DEEPER_MASK" in os.environ:
        for bottleneck in [50, 100, 200]:
            dirname = "hybrid_item_deeper_mask_c%i" % bottleneck
            net1_cfg = get_net((i_encoder1, u_encoder1),{"bottleneck":bottleneck, "code":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"item_mask"})
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=30, batch_size=128, shuffle=True, mean_centering=True,model_dir="/state/partition4/cbeckham/%s" % dirname,out_dir="output_new/%s" % dirname)
            
    if "HYBRID_USER_DEEPER" in os.environ:
        for bottleneck in [25,200]:
            dirname = "hybrid_user_deeper_c%i" % bottleneck
            net1_cfg = get_net((i_encoder1, u_encoder1),{"bottleneck":bottleneck, "code":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"user"})
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=30, batch_size=128, shuffle=True, mean_centering=True,model_dir="/state/partition4/cbeckham/%s" % dirname,out_dir="output_new/%s" % dirname)

    if "HYBRID_USER_DEEPER_MASK" in os.environ:
        for bottleneck in [50, 100, 200]:
            dirname = "hybrid_user_deeper_mask_c%i" % bottleneck
            net1_cfg = get_net((i_encoder1, u_encoder1),{"bottleneck":bottleneck, "code":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"user_mask"})
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=30, batch_size=128, shuffle=True, mean_centering=True,model_dir="/state/partition4/cbeckham/%s" % dirname,out_dir="output_new/%s" % dirname)


    # ------------------------
            
    if "HYBRID_BOTH_DEEPER_TIED_FIXED" in os.environ:
        for bottleneck in [25, 50, 100]:
            dirname = "hybrid_both_deeper_tied_fixed_c%i" % bottleneck
            net1_cfg = get_net(ui_encoder, {"bottleneck":bottleneck, "code":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"both"})
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=30, batch_size=128, shuffle=True, mean_centering=True, model_dir="/state/partition4/cbeckham/%s" % dirname, out_dir="output_new/%s" % dirname)

    if "HYBRID_BOTH_DEEPER_TIED_DOUBLY_FIXED" in os.environ:
        for bottleneck in [25, 50, 100]:
            dirname = "hybrid_both_deeper_tied_doubly_fixed_c%i" % bottleneck
            net1_cfg = get_net(ui_encoder_double, {"bottleneck":bottleneck, "code":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"both"})
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=30, batch_size=128, shuffle=True, mean_centering=True, model_dir="/state/partition4/cbeckham/%s" % dirname, out_dir="output_new/%s" % dirname)

    if "XX" in os.environ:
        for bottleneck in [25]:
            for k in [5, 10, 20]:
                dirname = "hybrid_both_lowrank_tied_fixed_k%i_c%i" % (k, bottleneck)
                net1_cfg = get_net(ui_encoder_lowrank1, {"bottleneck":bottleneck, "k":k, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"both"})
                train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=30, batch_size=128, shuffle=True, mean_centering=True, model_dir="/storeSSD/cbeckham/log6308_models/%s" % dirname, out_dir="output_new/%s" % dirname)
        

    if "HYBRID_BOTH_LOW_RANK_TIED" in os.environ:
        for seed in [0,1]:
            np.random.seed(seed)
            for bottleneck in [50]:
                for m1 in [30]:
                    dirname = "hybrid_both_lowrank_tied_fixed_m%i_c%i.%i" % (m1, bottleneck, seed)
                    net1_cfg = get_net(ui_encoder_lowrank, {"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"both", "m1":m1})
                    train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=15, batch_size=128, shuffle=True, mean_centering=True, model_dir="/storeSSD/cbeckham/log6308_models/%s" % dirname, out_dir="output_new/%s" % dirname)
            

    if "DEBUG" in os.environ:
        for bottleneck in [25]:
            dirname = "debug_c%i" % bottleneck
            net1_cfg = get_net(ui_encoder, {"bottleneck":bottleneck, "code":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid, "optimiser":"nesterov_momentum", "mode":"both"})
            train(net1_cfg, data=load_movielens10m_matrix_new(), num_epochs=30, batch_size=128, shuffle=True, mean_centering=True, model_dir="/state/partition4/cbeckham/%s" % dirname, out_dir="output_new/%s" % dirname)
