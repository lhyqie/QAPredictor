from numpy import genfromtxt
from numpy import vstack
import random
from crf import *


# The L2 regularization coefficient and learning rate for SGD
l2_coeff = 1
rate = 0.1

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

def train(A, XY, YY , maxiter = 100):
    w = [{} for x in range(len(XY))]
    for g in range(0, 4+1): # group id        
        w[g] = defaultdict(lambda: 0)
        for iternum in range(1, maxiter +1):
            #print 'iter ', iternum
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
                x = [] 
                x.append([0])
                for yy in YY[g]:
                    if yy in XY[g]:
                        x.append(XY[g][yy])
                    else:
                        x.append([])
                x.append([0]);
                #print x;
                my_grad, my_lik = calc_gradient(x, y, w[g], A[:,col])
                for k, v in my_grad.items(): grad[k] += v
                lik += my_lik
                #print my_lik
            l1 = sum( [abs(k) for k in grad.values()] )
            #print "Iter %r likelihood: lik=%r, reg=%r, reg+lik=%r gradL1=%r" % (iternum, lik, reg_lik, lik+reg_lik, l1)
            for k, v in grad.items(): w[g][k] +=  1.0  * v /l1 * rate     
                        
        #import operator
        #sorted_w = sorted(w[g].iteritems(), key= operator.itemgetter(1), reverse = True)
        #for e in sorted_w:
        #    print "%20s  %12s" % ( e[0], e[1])
    return w;


def generateAllYs(YY, g):
    yy = YY[g]  # [16,15,14]
    res = [];
    y = [0 for x in range(len(yy))];
    generateHelper(res, yy, y, 0)
    return res
    
def generateHelper(res, yy, y ,k):
    if k == len(yy):
        import copy
        res.append(copy.deepcopy(y))
    else:
        for i in range(1, 3+1):
            y[k] = i 
            generateHelper(res, yy, y, k+1)
            
# bx is a test point 
def test(bx, w, XY, YY):
    p = [{} for x in range(len(XY))]
    for g in range(0,4+1): # group id
        Ys = generateAllYs(YY, g)
        for y in Ys: # each possible y
            y = [0]+ y +[0]
            x = []
            x.append([0])
            for yy in YY[g]:
                if yy in XY[g]:
                    x.append(XY[g][yy])
                else:
                    x.append([])
            x.append([0])
            #print 'x = ', x
            #print 'y = ', y
            my_grad, my_lik = calc_gradient(x, y, w[g], bx)
            #print y , my_lik
            if len(y) == 3:
                p[g][y[1]] = my_lik;
            else: p[g][tuple(y[1:-1])] = my_lik
        
        #import operator
        #sorted_p = sorted(p[g].iteritems(), key= operator.itemgetter(1), reverse = True)
        #for e in sorted_p:
        #    print "%20s  %12s" % ( e[0], e[1])
    return p

if __name__ == '__main__':
    #print A
    #print BX
    #print BY
    
    n_features, n_instances = A.shape
    n_features -= 10
    n_folds = 5
    # 5-fold cross validation
    
    slots = range(n_instances)
    random.shuffle(slots)
    
    total_log_likelihood = 0
    n_tested = 0
    
    fold_size = n_instances / n_folds
    for fold in xrange(n_folds):
        print "fold %d" % fold 
        start = fold * fold_size
        if fold == n_folds-1:
            end = n_instances
        else:
            end = (fold+1)*fold_size
        test_subset = set(slots[start: end])
        train_subset = [k for k in slots if k not in test_subset]
        test_subset = list(test_subset)
        w = train(A[:, train_subset], XY, YY, 10)
        for i in xrange(len(test_subset)):
            log_likelihood = 0
            test_instance_id = test_subset[i]
            p = test(A[:,test_instance_id], w, XY, YY)
            for g in range(len(p)):
                # find the ground truth label for this group
                label_idx = [k-1 for k in YY[g]]
                if g < 3:
                    ground_label = int(A[label_idx, test_instance_id])
                else:
                    ground_label = tuple([int(k) for k in A[label_idx, test_instance_id]])
                log_likelihood += p[g][ground_label]
            total_log_likelihood += log_likelihood
            n_tested += 1
            print "neg-log-likelihood loss for the instance %d %f" % (test_instance_id, -log_likelihood)
            
        print "total loss %f" % -total_log_likelihood
        