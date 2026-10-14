import torch


def validate_cache(cache,grid_size=None):
    required={"images","occupancy","c2w","intrinsics_normalized","bounds","provenance"}
    if not required.issubset(cache):raise ValueError(f"Cache missing {required-set(cache)}")
    images,occ,poses=cache["images"],cache["occupancy"],cache["c2w"]
    if images.ndim!=4 or images.shape[1]!=3 or occ.ndim!=3 or poses.shape!=(len(images),4,4):
        raise ValueError("Cache shape mismatch")
    if grid_size is not None and occ.shape!=(grid_size,)*3:raise ValueError("Teacher grid and model grid differ")
    for key in ("images","occupancy","c2w","intrinsics_normalized","bounds"):
        if not torch.isfinite(cache[key]).all():raise ValueError(f"Nonfinite {key}")
    if (occ<0).any() or (occ>1).any() or (images<0).any() or (images>1).any():
        raise ValueError("Images/occupancy outside [0,1]")
    if cache["intrinsics_normalized"].shape!=(len(images),4) or (cache["intrinsics_normalized"][:,:2]<=0).any():
        raise ValueError("Invalid intrinsics")
    rotation=poses[:,:3,:3]
    if not torch.allclose(rotation.transpose(-1,-2)@rotation,torch.eye(3).expand_as(rotation),atol=1e-3) or not torch.allclose(torch.linalg.det(rotation),torch.ones(len(poses)),atol=1e-3):
        raise ValueError("Camera rotations must belong to SO(3)")
    if not torch.allclose(poses[:,3,:],torch.tensor([0.,0.,0.,1.]).expand(len(poses),-1),atol=1e-5):
        raise ValueError("Invalid homogeneous cameras")
    if not torch.allclose(poses[0],torch.eye(4),atol=1e-4):
        raise ValueError("Cache must be anchored to first camera; volume and poses need the same gauge")


