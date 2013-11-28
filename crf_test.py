from numpy import genfromtxt
from numpy import vstack
from collections import defaultdict
from math import log, exp
import sys
import operator

# The L2 regularization coefficient and learning rate for SGD
l2_coeff = 1
rate = 1


A = genfromtxt('quiztrain.csv', delimiter=',', skip_header = 0)
B = genfromtxt('quiztest.csv', delimiter=',', skip_header = 0)
BX = B[0:10,:]
BY = B[10:,:]

XY = [
         {12:[x for x in range(1, 11)]},
         {13:[x for x in range(1, 11)]},
         {20:[x for x in range(1, 11)]},
         {16:[5,9], 15:[2,3], 14:[2,4]},
         {},
     ]
YY = [
        [12],
        [13],
        [20],
        [16,15,14],
        [11,17,18,19]
      ]    

############# Utility functions ###################
def dot(A, B):
    return sum(A[k]*B[k] for k in A if k in B)
def add(A, B):
    C = defaultdict(A, lambda: 0)
    # for k, v in A.items(): C[k] += v
    for k, v in B.items(): C[k] += v
    return C
def logsumexp(A):
    k = max(A)
    return log(sum( exp(i-k) for i in A ))+k
 
############# Functions for memoized probability
def calc_feat(x, i, l, r, q):
    d = {}
    d["T", i, l, r] = 1  # Y-Y feature
    for x2 in x[i]:     # Y-X feature 
        d["E", i, x2, r, q[x2-1]] = 1
    return d

def calc_e(x, i, l, r, w, e_prob, q):
    if (i, l, r) not in e_prob:
        e_prob[i,l,r] = dot(calc_feat(x, i, l, r, q), w)
    return e_prob[i,l,r]

def calc_f(x, i, l, w, e, f, q):
    if (i, l) not in f:
        if i == 0:
            f[i,0] = 0
        else:
            prev_states = (range(1, 3+1) if i != 1 else [0])
            f[i,l] = logsumexp([
                calc_f(x, i-1, k, w, e, f, q) + calc_e(x, i, k, l, w, e, q)
                    for k in prev_states])
    return f[i,l]

def calc_b(x, i, r, w, e, b, q):
    if (i, r) not in b:
        if i == len(x)-1:
            b[i,0] = 0
        else:
            prev_states = (range(1, 3+1) if i != len(x)-2 else [0])
            b[i,r] = logsumexp([
                calc_b(x, i+1, k, w, e, b, q) + calc_e(x, i, r, k, w, e, q)
                    for k in prev_states])
    return b[i,r]
 
############# Function to calculate gradient ######
def calc_gradient(x, y, w, q):
    f_prob = {(0,0): 0}
    b_prob = {(len(x)-1,0): 0}
    e_prob = {}
    grad = defaultdict(lambda: 0)
    # Add the features for the numerator
    for i in range(1, len(x)):
        for k, v in calc_feat(x, i, y[i-1], y[i], q).items(): grad[k] += v
    # Calculate the likelihood and normalizing constant
    norm = calc_b(x, 0, 0, w, e_prob, b_prob, q)
    lik = dot(grad, w) - norm
    # Subtract the features for the denominator
    for i in range(1, len(x)):
        for l in (range(1, 3+1) if i != 1 else [0]):
            for r in (range(1, 3+1) if i != len(x)-1 else [0]):
                # Find the probability of using this path
                p = exp(calc_e(x, i, l, r, w, e_prob, q)
                        + calc_b(x, i,   r, w, e_prob, b_prob, q)
                        + calc_f(x, i-1, l, w, e_prob, f_prob, q)
                        - norm)
                # Subtract the expectation of the features
                for k, v in calc_feat(x, i, l, r, q).items(): grad[k] -= v * p
    # print grad
    # Return the gradient and likelihood
    return (grad, lik)

def train(A, XY, YY):
    w = [{} for x in range(len(XY))]
    # w[0] sihong 
    # w[1] sihong
    # w[2] sihong
    # w[3] w[4] huayi 
    # for 50 iterations
    g = 3 # group id 
    w[g] = defaultdict(lambda: 0)
    
    for iternum in range(1, 30+1):
        #print iternum
        grad = defaultdict(lambda: 0)
        # Perform regularization
        reg_lik = 0;
        for k, v in w[g].items():
            grad[k] -= 2*v*l2_coeff
            reg_lik -= v*v*l2_coeff
        # Get the gradients and likelihoods
        lik = 0
        for col in range(0, A.shape[1]):  # col is the index of train data  
            y = []
            y.append(0)
            for yInd in YY[g]:
                y.append(A[yInd-1,col])
            y.append(0)
            #print y;
            x = [[0]]
            for yy in YY[g]:
                x.append(XY[g][yy])
            x.append([0]);
            #print x;
            q = A[:,col]
            my_grad, my_lik = calc_gradient(x, y, w[g], q)
            for k, v in my_grad.items(): grad[k] += v
            lik += my_lik
        l1 = sum( [abs(k) for k in grad.values()] )
        print "Iter %r likelihood: lik=%r, reg=%r, reg+lik=%r gradL1=%r" % (iternum, lik, reg_lik, lik+reg_lik, l1)
        for k, v in grad.items(): w[g][k] += rate * v/l1  
    print w[g]
 
 
train(A, XY, YY)