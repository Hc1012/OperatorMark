"""Prospective numerical tests. Not discovered by unittest; UNEXECUTED.

Three exposed fixtures, one prescribed direction/cotangent each. No random draw.
Explicit future approval and existing owned supervision required by caller.
"""
from pathlib import Path
from msdev.records import write as _write,progress

def write(path,value):
    """Fsync raw diagnostics; nonfinite values remain explicit, not lost to JSON."""
    import math,os
    def encode(v):
        if isinstance(v,float) and not math.isfinite(v):return {'nonfinite':repr(v)}
        if isinstance(v,dict):return {k:encode(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [encode(x) for x in v]
        return v
    _write(path,encode(value))
    # Windows _commit/fsync requires a write-capable descriptor, even though
    # the committed JSON bytes are not changed by this flush.
    with Path(path).open('r+b') as f:os.fsync(f.fileno())

COUNTS=dict(P256_forward=3,P256_tangent=3,P256_tangent_continuation=3,P256_adjoint=6,P256_direct_oracle=12)

def run(out,check):
    import numpy as np
    from operatormark_numerics_v2.lock import FIXTURES
    from operatormark_numerics_v2.directions import feasible_direction
    from operatormark_numerics_v2.p1_family import P1DirectOracle,project_to_64
    from .p256_adjoint import Session,projection_transpose
    out=Path(out);out.mkdir(exist_ok=False);counts={k:0 for k in COUNTS};rows=[]
    def charge(k):check();counts[k]+=1;progress(out,'native_intent',counts=counts,operation=k)
    try:
        # Independent test of P^T, including boundary nodes and exact cell areas.
        n=256;nodal=np.linspace(-1,1,(n+1)**2,dtype=np.float64).reshape(n+1,n+1)
        w=np.cos(np.arange(4096,dtype=np.float64)).reshape(64,64)
        lhs=float(np.sum(project_to_64(nodal,n)[0]*w));rhs=float(np.sum(nodal*projection_transpose(w,n)))
        write(out/'PROJECTION.json',dict(lhs=lhs,rhs=rhs))
        # Raw operands survive even when the first assertion fails.
        def array(name,value):
            import os
            with (out/(name+'.npy')).open('xb') as f:
                np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())
        array('projection_nodal',nodal);array('projection_cotangent',w)
        array('projection_transpose',projection_transpose(w,n))
        if not np.isclose(lhs,rhs,rtol=1e-12,atol=1e-12):raise ValueError('independent exact-projection transpose identity')
        for name,rho in FIXTURES.items():
            v=feasible_direction(rho,np.arange(1,65,dtype=np.float64).reshape(8,8))
            charge('P256_forward')
            with Session(rho) as s:
                charge('P256_tangent');counts['P256_tangent_continuation']+=1
                t=s.base.tangent(v)
                array(name+'_rho',rho);array(name+'_direction',v);array(name+'_tangent',t.field64)
                write(out/(name+'_TANGENT.json'),dict(backward_error=float(t.tangent_backward_error)))
                if not np.isfinite(t.tangent_backward_error) or t.tangent_backward_error>1e-10:raise ValueError('frozen tangent gate')
                charge('P256_adjoint');g=s.vjp64(w);adjoint=s.last
                dot=float(np.sum(g*v));jdot=float(np.sum(w*t.field64))
                scale=max(abs(dot),abs(jdot),1e-12)
                array(name+'_vjp',g)
                write(out/(name+'_DUALITY.json'),dict(vjp_dot=dot,tangent_dot=jdot,scale=scale,adjoint=adjoint))
                if abs(dot-jdot)>1e-7*scale+1e-11:raise ValueError('frozen tangent / additive VJP duality')
                charge('P256_adjoint');zero=s.vjp64(np.zeros((64,64),dtype=np.float64))
                array(name+'_zero_cotangent',np.zeros((64,64),dtype=np.float64));array(name+'_zero_vjp',zero)
                write(out/(name+'_ZERO.json'),dict(nonzero=int(np.count_nonzero(zero)),adjoint=s.last))
                if np.count_nonzero(zero):raise ValueError('zero cotangent is not zero')
            # Frozen separately assembled sparse-LU oracle; second-order
            # one-sided differences stay feasible for binary boundary controls.
            oracle=P1DirectOracle();values={}
            for h in (0.,.0005,.001,.002):
                charge('P256_direct_oracle');field=oracle.field64(rho+h*v,256);values[h]=float(np.sum(w*field))
                array(name+'_direct_'+str(h),field)
                write(out/(name+'_DIRECT_'+str(h)+'.json'),dict(h=h,value=values[h]))
            fd1=(-3*values[0.]+4*values[.001]-values[.002])/.002
            fd2=(-3*values[0.]+4*values[.0005]-values[.001])/.001
            row=dict(fixture=name,vjp_dot=dot,tangent_dot=jdot,direct_slope_h1=fd1,direct_slope_h2=fd2,
                adjoint=adjoint,physical_qualification=False,thresholds_are_prospective_consistency_only=True)
            write(out/(name+'.json'),row);rows.append(row)
            if abs(fd2-dot)>2e-5*max(abs(dot),abs(fd2),1e-12)+1e-10:raise ValueError('independent direct-solve slope disagreement')
        if counts!=COUNTS:raise ValueError('native count reconciliation')
        write(out/'RESULT.json',dict(complete=True,counts=counts,rows=rows,physical_qualification=False))
    except BaseException as exc:
        write(out/'FAILURE.json',dict(complete=False,error=repr(exc),counts=counts,rows=rows));raise
