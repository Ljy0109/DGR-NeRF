# DGR-NeRF : Depth and Geometry Regularization for Neural Implicit Reconstruction From Few-View Satellite Images


# Installation
```
git clone https://github.com/Ljy0109/DGR-NeRF.git
cd DGR-NeRF

conda create -n dgr python=3.10.16 -y
conda activate dgr

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/centreborelli/sat-bundleadjust.git
cd sat-bundleadjust
pip install -e .
cd ..

git clone https://github.com/LiheYoung/Depth-Anything.git

pip install chardet==5.2.0 fire==0.7.1 kornia==0.8.1 plyflatten==0.2.0 pytorch_lightning==2.5.2 scikit_image==0.25.2 numba==0.61.0 trimesh==4.6.6 pyhocon==0.3.61 tensorboard pymcubes gradio_imageslider gradio opencv-python

# pip install torchsort==0.1.9 --no-build-isolation
pip install https://github.com/teddykoker/torchsort/releases/download/v0.1.10/torchsort-0.1.10+pt26cu118-cp310-cp310-linux_x86_64.whl

```

manually download the checkpoint : [depth-anything-large](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth)


load the model locally:
```
weight = torch.load("/root/mycode/DGR/Depth-Anything/weights/depth_anything_vitl14.pth")
```
# Running
```
chmod +x run_pipeline.sh
./run_pipeline.sh
```
