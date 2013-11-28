from numpy import genfromtxt

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
        [11,17,18,19],
      ]    

def train(A, XY, YY):
    w = [{} for x in range(len(XY))]
    # w[0] sihong 
    # w[1] sihong
    # w[2] sihong
    # w[3] w[4] huayi 
    return w;

# bx is a test point 
def test(bx, w, XY, YY):
    p = [{} for x in range(len(XY))]
    # p[0] sihong
    # p[1] sihong
    # p[2] sihong
    # p[3] p[4] xiaokai
    return p

print A
print BX
print BY