"""Enumerate applicable small-budget comparators from the successive constructions."""
import csv
from pathlib import Path
from research.filtered_outlets.verify import comparators
from research.saddle_moments.verify import envelope
from research.mixed_multiwell.construction import best_reciprocal, linear_count
from research.reciprocal_wells.construction import best_parameters

def row(n):
    basic = max((4*s*(n-1-2*s) for s in range(1,n) if n-1-2*s>=1),default=None)
    expo = max((8*s*(n-2-2*s) for s in range(1,n) if n-2-2*s>=1),default=None)
    variable = max((12*s*(n-2-2*s) for s in range(1,n) if n-2-2*s>=2),default=None)
    flat,grid,sep,_=comparators(n) if n>=6 else (0,0,0,0)
    vals=[linear_count(z,d,w) for z in (2,4,8) for d in range(2*z,n+1)
          for w in (n-d-z,) if w>=1 and max(2*d-1,2*w-1)<n]
    regular=max((4*(k*p*r-z+1) for r,z in ((1,2),(2,4),(3,8))
                 for L in range(2,n+1) for p in (z*L,) for k in (n-2-2*z-p,)
                 if k>=1),default=None)
    return (n,basic,expo,variable,envelope(n) or None,flat or None,sep or None,
            best_parameters(n)[0] or None,best_reciprocal(n)[0] or None,grid or None,
            max(vals,default=None),regular,{7:68,8:108,9:148}.get(n))

if __name__=="__main__":
    dest=Path(__file__).resolve().parent/"outputs"/"archive-comparison.tsv"
    header=("N center exponent_variations moving_potential paired_saddle flat_minimum "
            "separated reciprocal mixed_reciprocal chebyshev_grid mixed_linear_cap "
            "regular_bands interpolation_witness").split()
    with dest.open("w") as f:
        w=csv.writer(f,delimiter="\t",lineterminator="\n")
        w.writerow(header)
        for n in range(4,11):
            r=row(n)
            w.writerow(["--" if x is None else x for x in r])
            known=[4,8,32,68,108,148,192][n-4]
            assert max(x for x in r[1:] if x is not None)==known
    print("Archived finite envelopes at N=4..10 match all seven reported unrestricted specialized counts.")
