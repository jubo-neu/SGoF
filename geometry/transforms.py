import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6):
    """Two column vectors -> right-handed SO(3); degenerate vectors get a stable basis."""
    a1,a2=d6[...,:3],d6[...,3:]
    fallback=torch.zeros_like(a1); fallback[...,0]=1
    a1=torch.where(a1.norm(dim=-1,keepdim=True)<1e-6,fallback,a1)
    b1=F.normalize(a1,dim=-1)
    orthogonal=a2-(b1*a2).sum(dim=-1,keepdim=True)*b1
    axis=F.one_hot(b1.abs().argmin(dim=-1),3).to(b1.dtype)
    alt=axis-(axis*b1).sum(dim=-1,keepdim=True)*b1
    b2=F.normalize(torch.where(orthogonal.norm(dim=-1,keepdim=True)<1e-6,alt,orthogonal),dim=-1)
    b3=torch.cross(b1,b2,dim=-1)
    return torch.stack([b1,b2,b3],dim=-1)


def matrix_to_rotation_6d(matrix):
    return torch.cat([matrix[..., :,0],matrix[..., :,1]],dim=-1)


def pose9_to_c2w(pose):
    """9 parameters = 6D rotation + 3D translation (not 6 parameters total)."""
    output=torch.eye(4,device=pose.device,dtype=pose.dtype).expand(*pose.shape[:-1],4,4).clone()
    output[...,:3,:3]=rotation_6d_to_matrix(pose[...,:6])
    output[...,:3,3]=pose[...,6:9]
    return output


def c2w_to_pose9(c2w):
    return torch.cat([matrix_to_rotation_6d(c2w[...,:3,:3]),c2w[...,:3,3]],dim=-1)


def relative_to_first(c2w):
    return torch.linalg.inv(c2w[...,0:1,:,:])@c2w


def grid_centers(bounds,shape,device=None,dtype=torch.float32):
    """Tensor axes Z,Y,X; spatial vectors and bounds use X,Y,Z."""
    bounds=torch.as_tensor(bounds,device=device,dtype=dtype)
    if bounds.shape!=(2,3) or not torch.all(bounds[1]>bounds[0]):
        raise ValueError("bounds must be increasing [min_xyz,max_xyz]")
    z,y,x=shape
    coords=[bounds[0,i]+(torch.arange(n,device=device,dtype=dtype)+.5)*(bounds[1,i]-bounds[0,i])/n
            for i,n in enumerate((x,y,z))]
    zz,yy,xx=torch.meshgrid(coords[2],coords[1],coords[0],indexing="ij")
    return torch.stack([xx,yy,zz],dim=-1)


def fuse_features(points,features,depth,bounds,shape,chunk=65536):
    """Trilinear scatter with inverse-depth weights; positions in one anchored frame.

    points: (Q,3), features: (Q,C), depth: (Q,), shape=(Z,Y,X).
    Outside-grid points are excluded; missing cells remain zero and get zero support.
    """
    if points.ndim!=2 or features.ndim!=2 or points.shape[0]!=features.shape[0] or depth.shape!=(len(points),):
        raise ValueError("Invalid point/feature/depth shapes")
    b=torch.as_tensor(bounds,device=points.device,dtype=points.dtype)
    if b.shape!=(2,3) or not torch.all(b[1]>b[0]):
        raise ValueError("Invalid bounds")
    nz,ny,nx=shape
    dims=torch.tensor([nx,ny,nz],device=points.device,dtype=points.dtype)
    cells=nz*ny*nx
    numerator=torch.zeros(cells,features.shape[1],device=features.device,dtype=features.dtype)
    denominator=torch.zeros(cells,device=features.device,dtype=features.dtype)
    for start in range(0,len(points),chunk):
        p,f,d=points[start:start+chunk],features[start:start+chunk],depth[start:start+chunk]
        valid=(torch.isfinite(p).all(-1)&torch.isfinite(f).all(-1)&torch.isfinite(d)&(d>0)&(p>=b[0]).all(-1)&(p<b[1]).all(-1))
        p,f,d=p[valid],f[valid],d[valid]
        coords=(p-b[0])/(b[1]-b[0])*dims-.5
        floor=torch.floor(coords); fraction=coords-floor
        for dz in (0,1):
            for dy in (0,1):
                for dx in (0,1):
                    offset=torch.tensor([dx,dy,dz],device=p.device)
                    index=(floor.long()+offset).clamp_min(0)
                    index=torch.minimum(index,torch.tensor([nx-1,ny-1,nz-1],device=p.device))
                    w=torch.prod(torch.where(offset.bool(),fraction,1-fraction),dim=-1)/d.clamp_min(1e-6)
                    flat=index[:,2]*ny*nx+index[:,1]*nx+index[:,0]
                    numerator.index_add_(0,flat,f*w[:,None])
                    denominator.index_add_(0,flat,w)
    volume=(numerator/denominator.clamp_min(1e-8)[:,None]).reshape(nz,ny,nx,-1).permute(3,0,1,2)
    return volume,denominator.reshape(nz,ny,nx)
