from numpy import genfromtxt
from numpy import isnan
import numpy as np

def mapping(x):
    if x>=1 and x<=3:    #difficult
        return 1;
    elif x>=4 and x<=7:  #medium
        return 2;
    elif x>=8 and x<=10: #easy
        return 3;
    else:
        return -999;


# each row is a question
A1 = genfromtxt('quiztrain.csv', delimiter=',', skip_header = 0)
A2 = genfromtxt('quiztest.csv', delimiter=',', skip_header = 0)
A = np.concatenate( (A1,A2), axis = 1 );
print A1.shape
print A2.shape
print A.shape
 


from scipy.stats.stats import pearsonr

res = [];
for i in range(A.shape[0]):
    for j in range(i+1, A.shape[0]):
        res.append(['Q'+`i+1`,'Q'+`j+1`,pearsonr(A[i,:], A[j,:])[0]])
        

# sort correlation in the descending order of absolute value 
print '--------------------------------------------------';
res = sorted(res, key=lambda x: -abs(x[2]));

for e in res:
    print e;
