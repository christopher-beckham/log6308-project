import create_data
import numpy as np

X_train, _, _ = create_data.load_movielens10m_matrix_new()
nonzero = len( X_train.nonzero()[0] )
total = np.prod(X_train.shape)

print nonzero
print total
