import os
import json
import argparse

from PIL import Image
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from urllib.request import urlretrieve

from src.train import WSIModel, get_preprocessing_transforms

from tiatoolbox.wsicore.wsireader import WSIReader

class WSIParser:

    def __init__(self, batch_size = 16):
        weights_path = "models/weights.ckpt"

        if not os.path.exists(weights_path):
            url = "https://www.nextcloud.curlyspiker.fr/s/MEdBEQiwL7XH7y3/download/weights.ckpt"
            urlretrieve(url, weights_path)

        self.model = WSIModel.load_from_checkpoint(
            checkpoint_path="models/weights.ckpt",
            map_location=None,
        )
        self.model.eval()
        self.batch_size = batch_size

    def parse_slide_progress(self, path):
        """Opens a WSI slide and runs the CNN on 20x zoom patches to find nuclei areas. 

        This is a generator function in order to get intermediary results such as progress 
        percentage and partial mask, for a better user experience when the processing time 
        is too long.
        
        Args:
            - path: Path to the WSI tile (.ndpi file)

        Returns: 
            - progress: Float between 0 and 1 to indicate progression
            - mask: Numpy boolean array, at 1X zoom resolution, indicating nuclei presence
            - thumb: Numpy array reprensenting the RGB image of the slide at 1X zoom level
        """

        if not os.path.exists(path):
            print("Input file does not exist !")
            return None

        slide = WSIReader.open(path)
        tile_size = 256

        # Size of a 1x zoom image
        final_size = slide.slide_dimensions(1.0, 'power')

        # Size of a 20x zoom image
        target_res = 20
        image_size = slide.slide_dimensions(target_res, 'power')

        # Build the @D grid that corresponds to patches in image
        grid_sz = np.int32(np.array(image_size) / tile_size) + 1
        grid_indices = [(i,j) for i in range(grid_sz[0]) for j in range(grid_sz[1])]

        # Final mask that will receive information about nuclei presence
        mask = np.zeros(np.flip(grid_sz), dtype=bool)

        # Slide visualisation
        thumb = slide.slide_thumbnail(1.0)

        # Useful elements for ML pipeline
        transforms =  torchvision.transforms.Compose(get_preprocessing_transforms())
        batch_data = []

        with torch.no_grad():

            for idx, (i, j) in enumerate(grid_indices):

                # Extract the sub-image corresponding to the current patch
                bounds = [
                    i*tile_size,
                    j*tile_size,
                    (i+1)*tile_size,
                    (j+1)*tile_size,
                ]
                bounds = slide._bounds_at_resolution_to_baseline(bounds, resolution=target_res, units='power')
                img = slide.read_bounds(bounds=bounds, resolution=target_res, units='power')
                img = Image.fromarray(img)
                img = img.convert("RGB")

                # Do not process patches that are close to a full white image, to save some execution time
                if np.min(img) > 200:
                    continue

                # Append this patch to the curent batch
                batch_data.append(((i, j), transforms(img)))

                # When batch is complete, run it through the ML pipeline and update mask according to 
                # the resulting classification
                if len(batch_data) == self.batch_size or idx == len(grid_indices) - 1:
                    input_data = torch.stack([im for _, im in batch_data], dim=0).to(self.model.device)
                    res = self.model(input_data)

                    for label, data in zip(res, batch_data):
                        if label[0]<label[1]:
                            mask[data[0][1], data[0][0]] = 1

                    # Clear the batch after it has been processed
                    batch_data.clear()

                # Yield intermediary results to allow users to follow progression
                progress = idx / len(grid_indices)
                yield progress, np.array(Image.fromarray(mask).resize(final_size)), thumb
            
    def parse_slide(self, path):
        """Opens a WSI slide and runs the CNN on 20x zoom patches to find nuclei areas. 

        Args:
            - path: Path to the WSI tile (.ndpi file)

        Returns: 
            - mask: Numpy boolean array, at 1X zoom resolution, indicating nuclei presence
            - thumb: Numpy array reprensenting the RGB image of the slide at 1X zoom level
        """

        out = None
        print('Processing slide, this might take a while...')
        for _, mask, _ in self.parse_slide_progress(path):
            out = mask
        return out

def main():

    parser = argparse.ArgumentParser(
        prog='WSIParser',
        description='Find regions of interest in WSI slides, containing nuclei')

    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    parser.add_argument('-bs', '--batch_size', default=16)
    
    args = parser.parse_args()
    parser = WSIParser(args.batch_size)
    mask = parser.parse_slide(args.input)

    if mask is not None:
        Image.fromarray(mask).convert('RGB').save(args.output)

if __name__ == '__main__':
    main()

    

  
            