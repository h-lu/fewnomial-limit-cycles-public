"""Enclose the nine quartic signs for the BBJT transcription in inputs/public.json."""
import json
from pathlib import Path
from flint import arb, acb, ctx

data=json.loads(Path(__file__).with_name("inputs").joinpath("public.json").read_text())["bbjt"]
ctx.prec=160
a,b,c,d,e=map(arb,data["g"])
X,Y=arb("0.9"),arb("1.1")

def g(z,s):
    return a+b*z+c*z*s+d*z*z+e*s*s

def moment(h,large):
    t=arb(h).sqrt()
    if not large:
        # Circle in (x^2,y^2). The integrand fixes the orientation for the transcribed moment.
        def f(u,analytic):
            z=X+t*u.cos()
            s=Y+t*u.sin()
            return t*u.sin()*g(z,s)/(2*z.sqrt(analytic=analytic)*s.sqrt(analytic=analytic))
        limit=2*arb.pi()
    else:
        edge=(X+t).sqrt()
        def f(u,analytic):
            z=(X+t)*u.sin()**2
            w=edge*u.cos()*(t-X+z).sqrt(analytic=analytic)
            def sl(s):
                return g(z,s)/s.sqrt(analytic=analytic)
            return 2*edge*u.cos()*(sl(Y+w)-sl(Y-w))
        limit=arb.pi()/2
    v=acb.integral(f,0,limit,rel_tol=arb(2)**-80,abs_tol=arb(2)**-80,
                   deg_limit=100,eval_limit=160000,depth_limit=50)
    assert v.imag.contains(0) and (v.real>0 or v.real<0),(h,v)
    return v.real

if __name__=="__main__":
    for group,large in (("small",False),("large",True)):
        for h,sign in zip(data[group+"_h"],data[group+"_sign"]):
            v=moment(h,large)
            print(group,h,v.str(15),flush=True)
            assert (v>0)==(sign>0),(h,sign,v)
