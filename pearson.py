from numpy import genfromtxt
from numpy import isnan
A = genfromtxt('data.csv', delimiter=',', skip_header = 1)

def mapping(x):
    if x>=1 and x<=3: 
        return 3;
    elif x>=4 and x<=7:
        return 2;
    elif x>=8 and x<=10:
        return 1;
    else:
        return -999;

print len(A)
print A.shape
#print A[:,0]
#print A[:,1]

from scipy.stats.stats import pearsonr

#print A;
#for i in range(A.shape[0]):
#    for j in range(A.shape[1]):
#        A[i,j] = mapping(A[i,j])
#print A;

res = [];
for i in range(A.shape[1]):
    for j in range(i+1, A.shape[1]):
        #print i, j, pearsonr(A[:,i], A[:,j])[0]
        res.append([i+1,j+1,pearsonr(A[:,i], A[:,j])[0]])
        
for e in res:
    #print e
    if isnan(e[2]) : e[2] = 0; 
 
print '--------------------------------------------------';
res = sorted(res, key=lambda x: -abs(x[2]));

for e in res:
    print e;
