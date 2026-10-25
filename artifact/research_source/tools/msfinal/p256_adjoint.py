"""Additive development P1 adjoint; frozen discretization is imported unchanged.

Euclidean VJP of physical, exact-area-averaged 64x64 field wrt rho[8,8].
Derivative of the discrete SPD solution, NOT the finite PCG iteration program.
Not physically qualified; prospective tests remain unexecuted.
"""
def projection_transpose(cotangent,n):
    import numpy as np
    w=np.asarray(cotangent)
    if w.shape!=(64,64) or w.dtype!=np.float64 or not np.isfinite(w).all() or n not in (128,256):raise ValueError('physical float64 cotangent and reviewed level required')
    q=n//64;s=np.repeat(np.repeat(w,q,0),q,1)*(64**2/(6*n*n))
    nodal=np.zeros((n+1,n+1),dtype=np.float64)
    nodal[:-1,:-1]+=2*s;nodal[1:,1:]+=2*s
    nodal[:-1,1:]+=s;nodal[1:,:-1]+=s
    return nodal

class Session:
    def __init__(self,rho):
        from operatormark_numerics_v2.p1_family import P1Family
        self.base=P1Family(levels=(256,)).open(rho,256,compute_eta=False)
        self.closed=False;self.last=None
        if not self.base.forward.gates_pass:
            self.close();raise ValueError('P256 forward hard gate')
    @property
    def field64(self):return self.base.forward.field64.copy()
    def vjp64(self,cotangent):
        import dataclasses
        import numpy as np
        from operatormark.solvers import darcy_p1 as P
        if self.closed:raise RuntimeError('closed development P256 session')
        s=self.base.system
        # Benchmark boundary is fixed zero: no rho-dependent Dirichlet term.
        if np.any(s.boundary_values!=0):raise ValueError('only frozen zero-Dirichlet map is derived')
        rhs=projection_transpose(cotangent,256).reshape(-1)[s.mesh.interior_nodes]
        result=P.solve_p1_with_info(dataclasses.replace(s,reduced_rhs=rhs))
        lam=result.solution
        _,det,g=P._mesh_triangle_geometry(s.mesh)
        ulocal=self.base.nodal.reshape(-1)[s.mesh.triangles]
        llocal=lam.reshape(-1)[s.mesh.triangles]
        gu=np.einsum('tai,ta->ti',g,ulocal);gl=np.einsum('tai,ta->ti',g,llocal)
        triangle=-9*(det/2)*np.einsum('ti,ti->t',gu,gl)
        square=triangle.reshape(256*256,2).sum(axis=1).reshape(256,256)
        gradient=np.ascontiguousarray(square.reshape(8,32,8,32).sum(axis=(1,3)))
        if not np.isfinite(gradient).all():raise FloatingPointError('P256 adjoint gradient nonfinite')
        interior=lam.reshape(-1)[s.mesh.interior_nodes]
        self.last=dict(iterations=int(result.iterations),backward_error=float(P.p1_algebraic_backward_error(s.reduced_matrix,interior,rhs)),
                       differentiated_map='discrete physical projected field',physical_qualification=False)
        return gradient
    def close(self):
        self.base.close();self.closed=True
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
