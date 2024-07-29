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
conda activate afm_mdm4emp

# optional: verify correct installation of environment
conda env list
```
### Installation of (Foundation) Models
First of all, the following models have to be installed:
- Depth Anything v2
- Language Segment-Anything
- KISS-ICP
- GroundingDINO
- LightGlue


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

cd Depth-Anything-V2 && mkdir checkpoints
cd checkpoints && wget -O depth_anything_v2_vitl.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true

# Pre-trained Metric Depth Checkpoints

cd Depth-Anything-V2/metric_depth && mkdir checkpoints
cd checkpoints && wget -O depth_anything_v2_metric_vkitti_vitl.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth?download=true
```


### Language Segment Anything
Repeat the installation process for Language Segment Anything:
```
git clone https://github.com/luca-medeiros/lang-segment-anything && cd lang-segment-anything
pip install -e .
```

If the last command throws a pip dependency issue, ignore it. One possible solution lies in installing GroundingDINO manually:

```
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .
```
Once again, ignore any error messages. 
Download pre-trained model weights:
```
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```



### KISS-ICP
Installation of KISS-ICP can be done with
```
pip install kiss-icp
```

After following these steps, the resulting structure should look like this:

### LightGlue Feature Matching
```
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```
## Launching the Pipeline
The pipeline is launched with
```
python pipline.py
```

Since two foundation models are run at the same time (Depth Anything and Lang-SAM), memory requirements are high. For testing purposes, it could be interesting to generate masks in a separate step as seen in `demo_generate_masks.py` 

To facilitate imports, it is recommended to rename `Depth-Anything-V2` to `Depth_Anything_V2` and `lang-segment-anything` to `lang_segment_anything`. 

### Final Project Structure

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
├── LightGlue/
├── Depth-Anything-V2/           
    ├── checkpoints/
        ├── depth_anything_v2_vitl.pth   
    ├── metric_depth/
        ├── checkpoints/
            ├── depth_anything_v2_metric_vkitti_vitl.pth
```