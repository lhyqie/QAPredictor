from numpy import genfromtxt
from numpy import vstack
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

def train(A, XY, YY):
    w = [{} for x in range(len(XY))]
    # w[0] sihong 
    # w[1] sihong
    # w[2] sihong
    # w[3] w[4] huayi 
    # for 50 iterations
    g = 3 # group id
    # g = 4 
    w[g] = defaultdict(lambda: 0)
    for iternum in range(1, 30 +1):
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
        print "Iter %r likelihood: lik=%r, reg=%r, reg+lik=%r gradL1=%r" % (iternum, lik, reg_lik, lik+reg_lik, l1)
        for k, v in grad.items(): w[g][k] +=  1.0  * v /l1 * rate     
     
    import operator
    sorted_w = sorted(w[g].iteritems(), key= operator.itemgetter(1), reverse = True)
    for e in sorted_w:
        print "%20s  %12s" % ( e[0], e[1])




# bx is a test point 
def test(bx, w, XY, YY):
    p = [{} for x in range(len(XY))]
    # p[0] sihong
    # p[1] sihong
    # p[2] sihong
    # p[3] p[4] xiaokai
    return p

if __name__ == '__main__':
    #print A
    #print BX
    #print BY
    train(A, XY, YY)