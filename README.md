# Azure_kinect_alignment_with_CUDA

I did the alignment between color and depth image(Azure kinect dk) with CUDA. It takes so long time because it is necessary to tranform the 2D-3D conversion.
I tested this on Windows 10. If you wanna use this on Linux, you could compile this to nvcc without cmake (with dependency).

# The environment for testing is below.
- Azure kinect ( Color : [1280 x 720], Depth : [640 x 576])
- OS : Windows 10 (not Linux)
- IDE : Visual studio 2015 community
- CPU : Intel(R) Core(TM) i7-9700K (3.60GHz)
- GPU : Geforce RTX 2080 ti
- RAM : 64 GB

# Dependency
- Opencv : 4.3.0 (just for visualizing)
- Azure kinect SDK
- CUDA : 10.1

