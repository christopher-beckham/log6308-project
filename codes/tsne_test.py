import sys

import numpy as np
from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
from tsne import bh_sne


in_file = sys.argv[1]
out_file = sys.argv[2]
dat = np.load(in_file)

#model = TSNE(n_components=2, random_state=0, verbose=2)
#np.set_printoptions(suppress=True)
#transformed = model.fit_transform(dat)

transformed = bh_sne(dat)

np.save(out_file, transformed)
