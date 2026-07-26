import numpy as np 

M=np.array([[0,0,1,1], [1,0,0,0], [1,1,0,1], [1,1,0,0]])

d=0.85

col_sum=M.sum(axis=0)
M=M/col_sum
N=M.shape[0]

G=d*M + ((1-d)/N) * np.ones((N, N))

V=np.ones((N, 1))/N

def page_rank(G, V, iterations):
    for i in range(iterations):
        V=np.dot(G, V)
    return V

for k in [5, 10, 15]:
    V_K=page_rank(G, V, k)
    print(k)
    for i, val in enumerate(V_K):
        print(chr(i+65))
        print(val)