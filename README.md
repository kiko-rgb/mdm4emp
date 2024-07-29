# Mono Depth Maps for Ego Motion Prediction
This repository contains the final submission for the practical course 'Applied Foundation Model' at TUM during summer term 2024. Team Members are Jiarong Li, Konstantin Ikonomou, Miguel Trasobares Baselga. Supervising PhD candidate is Dominik Muhle.

## Getting Started

Create a project directory, and clone this repository with
```
git clone https://github.com/kiko-rgb/mdm4emp.git && cd mdm4emp
```

### Conda Environment
We recommend Conda to create an environment from the `environment.yml` file:
```
conda env create -f environment.yml
```
### Installing Required Models
First of all, the following models have to be installed:
- Depth Anything v2
- Language Segment-Anything
- KISS-ICP
- GroundingDINO (as a prerequisite)


### Depth Anything v2
Now install Depth Anything alongside this repository and install its requirements:
```
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
pip install -r requirements.txt
```

Depth Anything distinguishes between metric and relativ depth checkpoints. Therefore they are stored for convenience, for both variants:
```
# Relative Depth Checkpoints

cd Depth-Anything-V2/checkpoints
curl -O https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true

# Metric Depth Checkpoints

cd Depth-Anything-V2/metric_depth/checkpoints
curl -O https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth?download=true
```


### Language Segment Anything
Repeat the installation process for Language Segment Anything:
```
git clone https://github.com/luca-medeiros/lang-segment-anything && cd lang-segment-anything
pip install torch torchvision
pip install -e .
```

### KISS-ICP
Installation of KISS-ICP can be done with
```
pip install kiss-icp
```

After following these steps, the resulting structure should look like this:

```text
project/
├── mdm4emp/
    ├── afm_kiss_icp
    ├── config
    ├── ...
    ├── environnment.yml
    └── pipeline.py
    └── README.md
├── lang_segment_anything/
├── SAM_checkpoints/
├── GroundingDINO/
├── Depth_Anything_V2/           
    ├── checkpoints/
        ├── depth_anything_v2_vitl.pth   
    ├── metric_depth/
        ├── checkpoints/
            ├── depth_anything_v2_metric_vkitti_vitl.pth

```

## Launching the Pipeline
Launching the pipeline can then be done with
```
python pipline.py
```

Since two foundation models are run at the same time (Depth Anything and Lang-SAM), this step is quite memory intensive.