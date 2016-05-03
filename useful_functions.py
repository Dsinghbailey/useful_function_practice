import numpy as np





# softmax function
# gradient-log-normalizer of the categorical probability distribution.
def softmax(x):
    result=[]
    for j in x:
        result.append( np.exp(j)/sum([np.exp(i) for i in x]) )
    return np.asarray(result)


