from numpy import genfromtxt
from numpy import vstack
import random
import os
from crf import *


# The L2 regularization coefficient and learning rate for SGD
l2_coeff = 1
rate = 0.1

# A is matrix of training data, each column represents the answer of a student toward 20 questions
A = genfromtxt('quiztrain.csv', delimiter=',', skip_header = 0)

# model structures definition, XY is the dependency between X's and Y's and YY between Y's
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

"""
  input :  A -   training data
           XY -  dependency between X's and Y's
           YY -  dependency between Y's
           maxiter - max iteration number default 100
  output:  learned weights parameters
"""         
def train(A, XY, YY , maxiter = 100):
    w = [{} for x in range(len(XY))]
    for g in range(0, len(YY)): # group id        
        w[g] = defaultdict(lambda: 0)
        for iternum in range(1, maxiter +1):
            #print 'iter ', iternum
            grad = defaultdict(lambda: 0)
            # Perform regularization
            reg_lik = 0;
            for k, v in w[g].items():
                grad[k] -= 0#2*v*l2_coeff
                reg_lik -= 0#v*v*l2_coeff
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


"""
    for a YY[g] structure, generate all possible vector space YY[g] can take
"""
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
            
"""
    input:  bx - a vector that represents the answers of a student toward first 10 questions   
            w  - weight parameters learned from train()
            XY -  dependency between X's and Y's
            YY -  dependency between Y's
    output: probabilistic distribution of P(y|x) 
""" 
def test(bx, w, XY, YY):
    p = [{} for x in range(len(XY))]
    for g in range(0,len(YY)): # group id
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
            if len(YY[g]) == 1:
                p[g][y[1]] = my_lik;
            else: p[g][tuple(y[1:-1])] = my_lik
        
        #import operator
        #sorted_p = sorted(p[g].iteritems(), key= operator.itemgetter(1), reverse = True)
        #for e in sorted_p:
        #    print "%20s  %12s" % ( e[0], e[1])
    return p

"""
    subroutine for running five fold cross validation on training data
"""
def five_fold_cross_validation():
    n_features, n_instances = A.shape
    n_features -= 10
    n_folds = 5
    # 5-fold cross validation
    
    slots = range(n_instances)
    random.shuffle(slots)
    
    total_log_likelihood = 0
    fold_size = n_instances / n_folds
    for fold in xrange(n_folds):
        total_fold_log_likelihood = 0
        print "\n\nfold %d" % fold 
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
                if len(YY[g]) == 1:
                    ground_label = int(A[label_idx, test_instance_id])
                else:
                    ground_label = tuple([int(k) for k in A[label_idx, test_instance_id]])
                log_likelihood += p[g][ground_label]
                #print 'g = ', g , 'sum_exp()', sum([exp(x) for x in p[g].values()])
            total_fold_log_likelihood += log_likelihood
            total_log_likelihood += log_likelihood
            print "neg-log-likelihood loss for the Question %2d %20f" % (test_instance_id + 1, -log_likelihood)
            
        print " For this fold,  total log loss = %f , average log loss = %f " % (-total_fold_log_likelihood, -total_fold_log_likelihood/len(test_subset)) 
    
    print "For entire dataset, total log loss = %f , average log loss = %f " %(-total_log_likelihood, -total_log_likelihood/n_instances)


"""
    input:  the filepath of test data
    output: print to console the log loss of each data and average log loss
"""
def eval_test(testDataFilePath):
    B = genfromtxt(testDataFilePath, delimiter=',', skip_header = 0)
    n_features, n_instances = B.shape
    n_features -= 10
    total_log_likelihood = 0
    w = train(A, XY, YY, 10)
    for i in range(n_instances):
            log_likelihood = 0
            p = test(B[:,i], w, XY, YY)
            for g in range(len(p)):
                # find the ground truth label for this group
                label_idx = [k-1 for k in YY[g]]
                if len(YY[g]) == 1:
                    ground_label = int(A[label_idx, i])
                else:
                    ground_label = tuple([int(k) for k in A[label_idx, i]])
                log_likelihood += p[g][ground_label]
                #print 'g = ', g , 'sum_exp()', sum([exp(x) for x in p[g].values()])
            total_log_likelihood += log_likelihood
            print "neg-log-likelihood loss for the Question %2d %20f" % (i + 1, -log_likelihood)
            
    print "\n\nFor entire test dataset, total log loss = %f , average log loss = %f " %(-total_log_likelihood, -total_log_likelihood/n_instances)



######################################################################################################################
if __name__ == '__main__':
    optionMenu =  """
                     This is the team project for CS594 Fall 2013 with Prof. Brian Ziebart
                     Team members: XiaoKai Wei, Sihong Xie, Huayi Li 
                     
                     enter your option:
                            1. compute log loss using five-fold cross validation on the training dataset
                            2. compute log loss on the testing dataset 
                            3. exit
                  """
    while(True):
        option = input(optionMenu)
        if option == 1:
            five_fold_cross_validation()
        elif option == 2:
            testDataFilePath = raw_input("\nplease enter the path for the test data whose format is identical to quiztrain.csv\n")
            eval_test(testDataFilePath)
        elif option == 3:
            break;
        raw_input("\n\n press any key to continue...\n")
        os.system('cls')
        
    print '\n\n\n See you ....'
    