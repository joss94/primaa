import os
import json
import argparse

from PIL import Image
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from train import WSIModel, get_preprocessing_transforms

from tiatoolbox.wsicore.wsireader import WSIReader

class WSIParser:

    def __init__(self, batch_size = 16):
        self.model = WSIModel.load_from_checkpoint(
            checkpoint_path="models/weights.ckpt",
            map_location=None,
        )
        self.model.eval()
        self.batch_size = batch_size

    def parse_slide_progress(self, path):
        """TODO"""

        if not os.path.exists(path):
            print("Input file does not exist !")
            return None

        slide = WSIReader.open(path)
        target_res = 20
        tile_size = 256
        final_size = slide.slide_dimensions(1.0, 'power')
        image_size = slide.slide_dimensions(target_res, 'power')

        grid_sz = np.int32(np.array(image_size) / tile_size) + 1
        grid_indices = [(i,j) for i in range(grid_sz[0]) for j in range(grid_sz[1])]

        mask = np.zeros(np.flip(grid_sz), dtype=np.bool)
        transforms =  torchvision.transforms.Compose(get_preprocessing_transforms())
        batch_data = []

        thumb = slide.slide_thumbnail(1.0)

        with torch.no_grad():

            for idx, (i, j) in enumerate(grid_indices):
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

                if np.min(img) > 200:
                    continue

                batch_data.append(((i, j), transforms(img)))

                if len(batch_data) == self.batch_size or idx == len(grid_indices) - 1:
                    input_data = torch.stack([im for _, im in batch_data], dim=0).to(self.model.device)
                    res = self.model(input_data)

                    for label, data in zip(res, batch_data):
                        if label[0]<label[1]:
                            mask[data[0][1], data[0][0]] = 1
                    batch_data.clear()

                progress = idx / len(grid_indices)

                yield progress, np.array(Image.fromarray(mask).resize(final_size)), thumb
            
    def parse_slide(self, path):
        out = None
        print('Processing slide, this might take a while...')
        for progress, mask, thumb in self.parse_slide_progress(path):
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

    

  
            