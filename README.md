## Introduction

This is a repository containing code to segment regions of interest in WSI images for histopathology. 
It contains a model that has been train to detect patches that contain nuclei, in order to provide the user with a mask at 1X zoom level identifying regions of interest in the slide, for further analysis

## Installation

Because of the limitations on the size of the GitHub repositories, the weights are not included here. 
Pleases download them here (link to my self-hosted Nextcloud instance): https://www.nextcloud.curlyspiker.fr/s/tpxzpN6NXbzJHsb
Please copy the file to '''primaa/models/weights.ckpt'''

The application runs on Linux under Docker. 
To install, please run:

'''docker compose up'''

## How to use

The docker container will automatically start a container and open the port 8501 to access a streamlit application where the tool is ready to use. 
You can then access '''http://localhost:8501''' and upload a .ndpi file into the application.
The online app will display a thumbnail of the slide at the resolution 1X, and will progressively show the binary mask at the same time it is being calculate.

It is also possible to use a CLI tool:

First, find out the container name with 

'''docker ps -a'''

Then enter it with

'''docker exec -it <container-name> bash'''

Then call the '''wsi_parser''' CLI tool:

'''
cd /repo
python3 src/wsi_parser.py -i <path_to_ndpi> -o <path_to_png_output> -bs <batch_size>

Note that the application runs on CPU due to lack of information about the hardware it will run on. You can modify the Dockerfile for GPU-support in order to decreast the running time. 

## Training

This application uses a Resnet50 architecture trained on partially labeled data provided by Primaa. 
The weights are accessible within the repository and correspond to a final accuracy of 98.2% on a test set (10% of the data).

Training is done with PyTorch and PyTorch Lightning.

To reproduce those results, first make sure to mount the proper docker volume in the compose file:

'''
    volumes:
      - <data_dir>:/data
'''

where '''data_dir''' is a local directory containing the following structure 

'''
<data_dir>
    data
        data_nuclei_tiles
            labels.csv
            tile_0.png
            tile_1.png
            ...
'''

Then use the training script from inside the docker container: 

'''python3 src/train.py'''

The training script follows a weakly supervised learning approach by following these steps:
- Train on the labeled data (less than 50% of tiles)
- Use first version of model to automatically label remaining tiles
- Retrain with the whole dataset, mixing manual and automatic labels

The script automatically handles class imbalance by using weights in the binary cross entropy loss.

The training goes on for 20 epochs max, with an early stopping stratefy on the validation loss, with a patience of 5 epochs.

The batch size can be adjusted manually in the code if desired. 

You can track the training experiments in tensorboard (you will need to redirect the port 6006 in Docker):

'''
tensorboard --logdir=tb_logs/
'''

## Next steps

Missing improvements:

- Unit tests
- 