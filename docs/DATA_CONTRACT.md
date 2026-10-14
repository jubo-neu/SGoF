# Data and Cache Formats

Tensors are stored in local `.pt` files and loaded with `weights_only=True`. Images use RGB values in [0, 1]. Voxel axes are ordered Z, Y, X; spatial vectors are ordered X, Y, Z.

## Geometry Caches

- `features`: C, Z, Y, X; actual corresponding features from a frozen VGGT model.
- `support`: Z, Y, X; inverse-depth-weighted scatter support.
- `text_embedding`: a 512-dimensional embedding from a frozen CLIP model; the former demonstration configuration used 8 dimensions.
- `images`: N, 3, H, W; training context images.
- `c2w`: N, 4, 4; camera-to-world transforms, with the first camera set to the identity matrix.
- `intrinsics_normalized`: N, 4; ordered as fx/W, fy/H, cx/W, cy/H.
- `bounds`: 2, 3; finite spatial bounds in the first-camera reference frame.
- `voxel_colors`: optional 3, Z, Y, X fixed reference colors; neutral gray is used otherwise.
- `provenance`: includes `kind`, `scene_id`, model weight sources, and `uses_test_views=false`.

`tools/prepare_geometry_cache.py` reads point-aligned `points` (Q, 3), `features` (Q, C), `depth` (Q), camera transforms, and normalized intrinsics from an NPZ archive. Feature projection must use actual, traceable pretrained outputs. Do not mix coordinate scales or treat the world elevation axis as the camera depth axis.

## Student Targets

Required fields are `images`, `occupancy` (Z, Y, X), `c2w`, `intrinsics_normalized`, `bounds`, and `provenance`. The target volume and cameras must share the same reference frame. Validation rejects invalid rotations and a first-camera transform that is not the identity.

The training JSON contains a `samples` list. Each entry contains `id`, `scene`, `split` (`train` or `test`), and `cache` (a `.pt` path). A scene must not appear in both training and test splits. Create a separate manifest for each leave-one-scene-out fold. Samples within a batch must have the same number of views. Student training reads only `train` entries; text does not enter the student forward pass.

## Initialization Context

The context JSON contains `scene_id`, `images` (a list of RGB image paths), `bounds`, `uses_test_views=false`, and optionally `intrinsics_normalized`.

`bounds` is a predefined experimental convention and must not be selected from held-out reference geometry. Providing intrinsics constitutes known calibration, rather than strictly RGB-only input. Omitting intrinsics requires training the optional prediction head with `lambda_intrinsics > 0` and reporting it as an extension.

## Rendering Evaluation CSV

Required fields are `method`, `scene`, `view`, `seed`, `protocol`, `prediction`, and `target`. Optional fields are `sky_mask`, `far_mask`, `boundary_mask`, and `category`.

Relative paths are resolved against the CSV location. All methods and seeds within a protocol must use identical scene/view sets. Image dimensions are not silently resampled. Empty masks do not receive a score of zero. Do not fabricate rows for missing runs; record failures separately and retain them when reporting completion rates.

Unified LPIPS evaluation uses the official AlexNet model; the value is left empty if its dependencies are unavailable. Nonzero mask pixels define the evaluation region. Masks for finite far geometry must come from independent geometry, rather than the evaluated method's predicted depth.
