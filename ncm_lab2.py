import numpy as np

def matrix_toFixed(numObj, digits=0):
    for i in range(numObj.shape[0]):
        for j in range(numObj.shape[1]):
           numObj[i,j]=f"{numObj[i,j]:.{digits}f}"
    return numObj

def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"

def gsum(i,S,x):
    res=0
    for j in range(S.shape[0]):
        res+=(S[i,j]*x[j,0])
    return res

def matrix_rate(A):
    x=np.zeros([A.shape[0],1])
    for i in range(A.shape[0]):
        res=0
        for j in range(A.shape[0]):
            res+=abs(A[i,j])
        x[i,0]=res
    return x.max()

def cond(A):
    return (matrix_rate(A)*matrix_rate(A.I))

def gaus_forward(S,b):
    sb = np.hstack((S,b))
    n = S.shape[0]
    x = np.zeros([b.shape[0], b.shape[1]])
    i = n-1
    while i >= 0:
        x[i, 0] = (sb[i, n] - gsum(i, S, x)) / S[i, i]
        i -= 1
    return x

def gaus_backward(S, b):
    sb = np.hstack((S,b))
    n = S.shape[0]
    x = np.zeros([b.shape[0], b.shape[1]])
    for i in range(n):
        x[i,0] = (sb[i, n]-gsum(i, S, x))/S[i, i]
    return x

def summ(i,j,S,D):
    res = 0
    for p in range(i):
        res += S[p,i]*D[p,p]*S[p,j]
    return res

def square_roots(A,b):
    S = np.matrix(np.zeros([A.shape[0],A.shape[1]]))
    D = np.matrix(np.zeros([A.shape[0], A.shape[1]]))
    detA = 1
    if (A != A.T).any():
        raise ValueError
    S[0,0] = (abs(A[0,0]))**(1/2)
    D[0,0] = np.sign(A[0,0])
    S[0,1] = A[0,1]/(S[0,0]*D[0,0])
    for i in range(1,A.shape[0]):
        S[i, i] = (abs(A[i, i] - summ(i, i, S, D))) ** (1 / 2)
        D[i, i] = np.sign(A[i, i] - summ(i, i, S, D))
        for j in range(i+1,A.shape[1]):
            S[0, j] = A[0, j] / (S[0, 0] * D[0, 0])
            S[i,j] = (A[i,j]-summ(i,j,S,D))/(S[i,i]*D[i,i])
    y = gaus_backward(S.transpose()*D,b)
    x = gaus_forward(S,y)
    for i in range(A.shape[0]):
        detA *= D[i,i]*((S[i,i])**2)
    return [x, detA]

def jsum(i, A, x):
    res=0
    for j in range(A.shape[0]):
        if i==j:
            continue
        res+= A[i,j]*x[j,0]
    return res

def jsumm(A):
    res=[]
    for i in range(A.shape[0]):
        a=0
        for j in range(A.shape[0]):
            if i == j:
                continue
            a+=abs(A[i,j])
        res.append(a)
    return res

def jacobi(A,b):
    x = np.matrix(np.ones([b.shape[0], b.shape[1]]))
    x0=np.matrix(np.zeros([b.shape[0], b.shape[1]]))
    eps=10**(-4)
    for i in range(A.shape[0]):
        if abs(A[i,i]) < jsumm(A)[i]:
            raise TypeError
    while ((abs(x0-x)).max()) >= eps:
        for i in range(A.shape[0]):
            x[i,0]=x0[i,0]
            x0[i,0]=(b[i,0]-jsum(i,A,x0))/A[i,i]
    return x0

def output(s):
    with open("outputs.txt","w") as out_file:
        out_file.write(s)

Ab = []
A=[]
B=[]
n=0
with open("inputs.txt") as file:
    for line in file:
        if "#" in line[0] or " " in line[0]:
            continue
        else:
            if "n=" in line[0:2]:
                n = int(line[2:])
            else:
                Ab.append([float(i) for i in line.split()])
                A.append([float(i) for i in line.split()[0:n]])
                B.append([float(i) for i in line.split()[n:n+1]])

ab = np.matrix(Ab)
a = np.matrix(A)
b = np.matrix(B)
out = ""
er1 = "Method squad roots:\nError!\nMatrix must be symmetrical\n\n"
er2 = "Method Jacobi:\nError!\nThe condition of diagonal superiority isn`t fulfilled\n\n"
try:
    res = square_roots(a, b)
    out += "Method squad roots:\nx={0}\n\ndetA= {1}\n\n".format(matrix_toFixed(res[0], 4), toFixed(res[1], 4))
except(ValueError):
    out += er1
try:
    res2 = jacobi(a, b)
    out += "Method Jacobi:\nx={0}\n\n".format(matrix_toFixed(res2, 4))
except(TypeError):
    out += er2
finally:
    out += "cond(A)= {0}".format(toFixed(cond(a), 4))
    output(out)
