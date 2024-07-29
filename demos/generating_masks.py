import numpy as np
import torch, torchvision
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
import cv2
import matplotlib
matplotlib.use('agg')  # Or any other X11 back-end
import matplotlib.pyplot as plt

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Go two levels up
parent_dir = os.path.dirname(module_path)
# Add the parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from PIL import Image
from lang_sam import LangSAM
from mdmemp.datasets.afm_kitti_360 import AFMKITTI360Dataset

"""
This scripts generates masks for the sky in the KITTI-360 dataset using the Lang-Segment Anything model.
The masks are saved in the /data/masks/ directory according to its sequence number.

The values of the masks are TRUE for the sky and FALSE for the rest of the image. Consider this when using the masks.
"""

dataset_path = '/storage/group/dataset_mirrors/01_incoming/kitti_360/KITTI-360'
poses_path = '/storage/group/dataset_mirrors/01_incoming/kitti_360/KITTI-360/data_poses'
config_file = '../config/KITTI_config.yaml'
sequences_ids = ['0000', '0002', '0003', '0004', '0005', '0006', '0007', '0009', '0010']

# load LANG-Segment Anything model
model = LangSAM(sam_type="vit_l")
text_prompt = "sky"

def main():
    for seq_id in sequences_ids:
        print(f"### Processing sequence {seq_id}...")
        kitti360_sequence = AFMKITTI360Dataset(dataset_path, poses_path, sequence_id=seq_id, split_path=None, return_scans=True, return_stereo=True, target_image_size=(192, 640))
        num_images = len(kitti360_sequence._img_ids["2013_05_28_drive_" + seq_id + "_sync"])

        # output masks to @miguel folder
        sequence_mask_output_path = '/data/masks/' + seq_id + '/'
        os.makedirs(sequence_mask_output_path, exist_ok=True)

        num_no_masks = 0
        for i in range(num_images):
            item = kitti360_sequence.__getitem__(i)
                    
            if "imgs" in item and len(item["imgs"]) > 0:  # Check if "imgs" exists and has at least one item
                # take left image
                img_l = item["imgs"][0]
                # convert to PIL image
                image_pil = Image.fromarray((img_l*255).astype(np.uint8))

                # get mask from LANG_SAM
                masks, _, _, _ = model.predict(image_pil, text_prompt)
                # convert mask to numpy
                masks_np = [mask.squeeze().cpu().numpy() for mask in masks]

                if masks_np:
                    # save them in your folder
                    mask_name = str(i).zfill(10)
                    mask_path = f"{sequence_mask_output_path}{mask_name}.npz"
                    np.savez_compressed(mask_path, x=masks_np[0])
                else:
                    print(f"No mask found for image {i}!")
                    num_no_masks += 1
                    
        print(f"{i} images were processed, a sky was detected in {(i-num_no_masks)/i} percent of images.")
        print("Number of images without masks: ", num_no_masks, "corresponding to ", num_no_masks/i, " percent of the images.")
        
        # command to unpack the masks:
        '''
        def get_mask(index, sequence_id):
            mask_output_path = f'/data/masks/{sequence_id}/'
            sample_mask = str(index).zfill(10)
            mask_path = f"{mask_output_path}{sample_mask}.npz"
            return np.load(mask_path)['x']

        item = kitti360_sequence.__getitem__(i)

        if "imgs" in item and len(item["imgs"]) > 0:  # Check if "imgs" exists and has at least one item
            # take left image
            left_image = item["imgs"][0]
            # get mask
            mask = get_mask(index=frame_id, sequence_id=seq)
            masked_image_array = np.zeros_like(left_image)
            masked_image_array[~mask] = left_image[~mask]
            
            plt.imshow(masked_image_array)
        '''

if __name__ == "__main__":
    print("Generating masks for the sky in the KITTI-360 dataset using Lang-Segment Anything model.")
    main()