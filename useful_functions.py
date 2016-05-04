import numpy as np





# softmax function
# gradient-log-normalizer of the categorical probability distribution.
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


