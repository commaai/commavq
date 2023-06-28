import numpy as np

def multinomial(prob_matrix):
  prob_matrix /= prob_matrix.sum(axis=1, keepdims=True)
  s = prob_matrix.cumsum(axis=1)
  r = np.random.rand(prob_matrix.shape[0])
  k = (s < r).sum(axis=1)
  return np.expand_dims(k, -1)

def softmax(x, axis=None):
  x = x - x.max(axis=axis, keepdims=True)
  y = np.exp(x)
  return y / y.sum(axis=axis, keepdims=True)
