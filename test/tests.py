import sys
sys.path.append('/Users/matt/OneDrive/ucsf/algorithms/final')
from ann_class import NNet
import numpy as np

def test_identity_matrix():
    autoencoder = NNet(8,3,8)
    X = np.identity(8)
    y = np.identity(8)
    autoencoder.train(X, y, 10000)
    assert X.all() == np.rint( autoencoder.predict(X) ).all()
    #return autoencoder.predict(X)
test_identity_matrix()
