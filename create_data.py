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


################
# data loading #
################

def load_movielens10m_matrix():
    data_dir = "data/movielens10m"
    ratings = np.load("%s/movielens10m.pkl_01.npy" % data_dir)
    movies = np.load("%s/movielens10m.pkl_02.npy" % data_dir)
    users = np.load("%s/movielens10m.pkl_03.npy" % data_dir)
    tmp = csr_matrix((ratings, (users, movies)))
    return tmp

def iterator(mat, bs=32, mean_centering=False, shuffle=False):
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
        if mean_centering:
            for i in range(0, this_batch.shape[0]):
                row = this_batch[i]
                #pdb.set_trace()
                row_mean = np.sum(row) / ((row != 0).sum()+1)
                row = row - row_mean
                this_batch[i] = row
        yield this_batch, (this_batch != 0).astype("float32")
        b += 1

def prep_data(mean_center=False):
    # load in the data, then shuffle it randomly
    X = load_movielens10m_matrix()
    idxs = [x for x in range(0, X.shape[0])]
    X = X[idxs]
    # split into train set, valid set, and test set
    n = X.shape[0]
    X_train, X_valid, X_test = X[0:int(0.8*n)], X[int(0.8*n):int(0.9*n)], X[int(0.9*n)::]
    return X_train, X_valid, X_test

####################
# lasagne networks #
####################

def i_encoder1(args):
    n_items = 65133
    l_in = InputLayer((None, n_items))
    l_dense = DenseLayer(l_in, num_units=args["bottleneck"])
    l_dense2 = DenseLayer(l_dense, num_units=args["code"])
    l_inv = l_dense2
    for layer in get_all_layers(l_inv)[::-1]:
        if isinstance(layer, InputLayer):
            break
        l_inv = InverseLayer(l_inv, layer)
    return l_inv

def i_encoder1_no_tieing(args):
    n_items = 65133
    l_in = InputLayer((None, n_items))
    l_dense = DenseLayer(l_in, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"])
    l_dense2 = DenseLayer(l_dense, num_units=args["code"], nonlinearity=args["nonlinearity"])
    l_inv = DenseLayer(l_dense2, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"])
    l_inv = DenseLayer(l_inv, num_units=n_items, nonlinearity=args["nonlinearity"])
    return l_inv

def simple_net(args):
    n_items = 65133
    l_in = InputLayer((None, n_items))
    l_dense = DenseLayer(l_in, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"])
    l_inv = DenseLayer(l_dense, num_units=n_items, nonlinearity=linear)
    return l_inv

def simple_net_tieing(args):
    n_items = 65133
    l_in = InputLayer((None, n_items))
    l_dense = DenseLayer(l_in, num_units=args["bottleneck"], nonlinearity=args["nonlinearity"])
    l_inv = InverseLayer(l_dense, l_dense)
    return l_inv


# ################
# lasagne helper #
# ################

def get_net(net_fn, args):
    print "compiling theano functions..."
    X = T.fmatrix('X')
    M = T.fmatrix('M') # mask
    l_out = net_fn(args)
    print_network(l_out)
    net_out = get_output(l_out, X)
    if "mask" in args:
        print "masking mode = true"
        loss = (M*squared_error(net_out, X)).mean()
    else:
        loss = squared_error(net_out, X).mean()
    params = get_all_params(l_out, trainable=True)
    lr = theano.shared(floatX(args["learning_rate"]))
    updates = nesterov_momentum(loss, params, learning_rate=lr, momentum=0.9)
    train_fn = theano.function([X,M], loss, updates=updates, on_unused_input='warn')
    loss_fn = theano.function([X,M], loss, on_unused_input='warn')
    out_fn = theano.function([X], net_out)
    return {
        "l_out": l_out,
        "train_fn": train_fn,
        "loss_fn": loss_fn,
        "lr": lr,
        "out_fn", out_fn
    }

def print_network(l_out):
    for layer in get_all_layers(l_out):
        print layer, layer.output_shape
    print "number of params: %i" % count_params(layer)

def test(net_cfg, X_test, mean_centering=False):
    # mean center the test set
    row_means = []
    if mean_centering:
        for i in range(0, X_test.shape[0]):
            row = X_test[i]
            #pdb.set_trace()
            row_mean = np.sum(row) / ((row != 0).sum()+1)
            row = row - row_mean
            X_test[i] = row
            row_means.append([row_mean])
    # this is a column vector of means
    row_means = np.asarray(row_means, dtype="float32")
    # get the predictions
    out_fn = net_cfg["out_fn"]
    X_test_reconstruction = out_fn(X_test)
    # add the row means back in

    pass
    
def train(net_cfg, data, num_epochs, batch_size, out_dir, schedule={}, resume=None, shuffle=False, mean_centering=False, quick_check=False):
    print "training..."
    #pdb.set_trace()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    headers = ["epoch", "train_loss", "valid_loss", "time"]
    l_out, train_fn, loss_fn, lr = net_cfg["l_out"], net_cfg["train_fn"], net_cfg["loss_fn"], net_cfg["lr"]
    if resume != None:
        write_flag = "ab"
        print "resuming training..."
        with open(resume) as g:
            set_all_param_values(l_out, pickle.load(g))
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
        train_losses = []
        for X_batch, X_mask in iterator(X_train, bs=batch_size, shuffle=shuffle, mean_centering=mean_centering):
            #pdb.set_trace()
            this_loss = train_fn(X_batch, X_mask)
            f_train_raw.write("%f\n" % this_loss); f_train_raw.flush()
            train_losses.append(this_loss)
            if quick_check:
                break
        avg_train_loss = np.mean(train_losses)
        valid_losses = []
        for X_batch, X_mask in iterator(X_valid, bs=batch_size, shuffle=False, mean_centering=mean_centering):
            this_loss = loss_fn(X_batch, X_mask)
            f_valid_raw.write("%f\n" % this_loss); f_valid_raw.flush()
            valid_losses.append(this_loss)
            if quick_check:
                break
        avg_valid_loss = np.mean(valid_losses)
        out_str = "%i,%f,%f,%f\n" % (epoch+1, avg_train_loss, avg_valid_loss, time()-t0)
        f_clean.write("%s\n" % out_str)
        f_clean.flush()
        print out_str
        with open("%s/%i.model" % (out_dir, epoch+1), "wb") as g:
            pickle.dump( get_all_param_values(l_out), g, pickle.HIGHEST_PROTOCOL)
        

if __name__ == '__main__':

    np.random.seed(0)

    X_train, X_valid, X_test = prep_data()
    
    print X_train.shape
    print X_valid.shape
    print X_test.shape

    #for x_batch in iterator(mat):
    #    print x_batch.shape

    #net1 = i_encoder1({"bottleneck":200, "code":200})
    #print_network(net1)
    
    #net1_cfg = get_net(simple_net, {"bottleneck":500, "learning_rate":0.1, "nonlinearity":sigmoid})
    #train(net1_cfg, data=(X_train,X_valid,X_test), num_epochs=10, batch_size=128, shuffle=True, mean_centering=True, out_dir="output/simple_net")
    
    #net1_cfg = get_net(simple_net, {"bottleneck":500, "learning_rate":0.01, "nonlinearity":sigmoid})
    #train(net1_cfg, data=(X_train,X_valid,X_test), num_epochs=10, batch_size=128, shuffle=True, mean_centering=True, out_dir="output/simple_net_lr0.01")
    
    #net1_cfg = get_net(simple_net_tieing, {"bottleneck":500, "learning_rate":0.1, "nonlinearity":sigmoid})
    #train(net1_cfg, data=(X_train,X_valid,X_test), num_epochs=10, batch_size=128, shuffle=True, mean_centering=True, out_dir="output/simple_net_tieing")

    #bottleneck=200
    for bottleneck in [50, 100, 300, 400, 500]:
        net1_cfg = get_net(simple_net, {"bottleneck":bottleneck, "learning_rate":0.1, "nonlinearity":sigmoid})
        train(net1_cfg, data=(X_train,X_valid,X_test), num_epochs=10, batch_size=128, shuffle=True, mean_centering=True, out_dir="output/simple_net_c%i" % bottleneck)


    

    
