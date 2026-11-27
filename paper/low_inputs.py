"""Construct exact component supports with symbolic admissible parameters, N=1..10."""
import json
from math import comb
from pathlib import Path
import sympy as s

X,Y=s.symbols("x y")
A,mu,z,gamma,q=s.symbols("A mu delta gamma q")
R=s.Rational
def collect(P,Q):
    rows=[]
    for poly in (P,Q):
        rows.append([dict(x=i,y=j,c=str(s.factor(c))) for (i,j),c in s.Poly(poly,X,Y).terms() if c!=0])
    return dict(P=rows[0],Q=rows[1],support=list(map(len,rows)),
                D=max(r["x"]+r["y"] for part in rows for r in part))

def base(N):
    if N<=3:
        return collect(1,(0,1,1+X)[N-1])
    n=(N-1)//2
    t=N-1-n
    h=s.Symbol("h")
    roots=s.Poly(s.prod(h-j for j in range(1,t)),h)
    G=sum(roots.nth(j)*4**j*(j+1)/s.binomial(2*j,j)*Y**(2*j) for j in range(t))
    return collect(2*Y,-s.diff(A**2*s.chebyshevt(n,X)**2,X)+mu*s.integrate(G,Y))

def well(n,d):
    h=s.Symbol("h")
    roots=s.Poly(s.prod(h-i for i in range(1,d)),h)
    def kj(j):
        return R((2*j+1)*comb(2*j,j),4**j*(j+1))
    F=gamma*Y
    for j in range(d):
        L=sum(comb(2*j+1,2*k)*Y**(2*j+1-2*k)*(2-Y**2)**k for k in range(j+1))
        F+=kj(d-1)/kj(j)*z**(d-1-j)*roots.nth(j)*L
    return collect(4*Y**3-4*Y,-s.diff(A**2*s.chebyshevt(n,X)**2,X)+mu*F)

def turn(n):
    # d=4, r=7, b=2. Ratios of beta values in S3's two analytic rows.
    r,b,d=7,2,4
    def ratios(offset,beta):
        raw=[s.Integer(1)]
        for j in range(d):
            raw.append(raw[-1]*(j+offset)/(j+offset+beta))
        return [t/raw[b] for t in raw]
    R0=ratios(R(1,2)+R(r,4),R(3,2))
    R1=ratios(R(1,2)-R(r,4),R(1,2))
    def control(a,b0):
        cb=(b0-a)/(R1[d]-R0[d])
        return (a-R0[d]*cb)*Y**4+cb*Y**8
    vectors=[(0,control(1,0)),(9,1+control(-R0[0],-R1[0])),
             (13,Y**2+control(-R0[1],-R1[1])),(14,control(0,1))]
    G=s.Poly(sum((-1)**i * q**((32-e)**2)*v for i,(e,v) in enumerate(vectors)),Y)
    # I_d = r/(r+2) I_{d-1}; substitute before taking the large A limit.
    density=G.as_expr()-G.nth(8)*Y**8+R(7,9)*G.nth(8)*Y**6
    F=s.integrate(density,Y)
    return collect(9*Y**8-7*Y**6,-s.diff(A**2*s.chebyshevt(n,X)**2,X)+A*mu*F)

def focus(k,both):
    a=-1-z**4/24 if k==1 else -1+z**4/6
    b=0 if k==1 else 5*z**2/54
    if both:
        return collect(Y*(a*X**2*Y**4+b*Y**6-a-b),X*(X**2*Y**2-1))
    return collect(a*X**2*Y**2+b*Y**3-a-b,2*X*(X**2*Y-1))

if __name__=="__main__":
    res={}
    for N in range(1,11):
        cap=(base(N) if N<=4 else focus(2,False) if N==5 else
             well(2,N-4) if N in (6,7,8) else turn(N-6))
        elementary=base(N)
        for row in (cap,elementary):
            assert sum(row["support"])==N and row["D"]<N
        res[N]={"elementary":elementary, "uniform":cap if N >= 9 else elementary, "capped_specialized":cap}
        if N in (4,5):
            ref=focus(N-3,True)
            assert sum(ref["support"])==N and ref["D"]==7
            res[N]["two_squares"]=ref
    print(json.dumps(res,indent=2))
