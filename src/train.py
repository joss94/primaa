from pathlib import Path
import os
import re
import time

from tqdm import tqdm
import pandas as pd
from PIL import Image
import lightning as L
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics.classification import BinaryAccuracy

def get_preprocessing_transforms():
    """
    Returns the transformations to apply to all images, including during inference
    Calling this method ensures consistency in the image preprocessing between training
    and inference. 
    """
    return [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=0, std=1),
    ]



class WSIDataset(data.Dataset):
    """Class responsible for building a dataset from the CSV labels and the PNG tiless"""

    def __init__(self, data_dir, labels_filename):
        """
        Builds a dataset from PNG tiles and CSV labels
        """

        super().__init__()

        self.data_points = []

        # Read all labels from CSV, and link it to its tile filepath if it exists
        annots = pd.read_csv(os.path.join(data_dir, labels_filename))
        for index, row in annots.iterrows():
            filepath = os.path.join(data_dir, f'tile_{row["tile_id"]}.png')
            if not os.path.exists(filepath):
                print(f"Did not find {filepath}")
                continue
            nuclei = 0 if row['label'] == "NO_NUCLEUS" else 1
            self.data_points.append((filepath, torch.FloatTensor([1-nuclei, nuclei])))


        # Build the transformations that combine a preprocessing step (normalization)
        # and a data augmentation (rotation)
        transforms = get_preprocessing_transforms()
        transforms.insert(0, torchvision.transforms.RandomRotation(degrees=180))
        self.transforms = torchvision.transforms.Compose(transforms)

        # Compute the amount of data points in each class, it will be useful to handle class imbalance
        # with a weighted loss
        self.class_rep = annots['label'].value_counts()

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        """Returns a pair for each data point, made of a tensor representation the tile
        and its matching label (NUCLEI is 1, NO_NUCLEUS is 0)"""
        path, label = self.data_points[idx]

        with open(path, "rb") as f: 
            img = Image.open(f) 
            img = img.convert("RGB")
            img = self.transforms(img)

        return img, label

class WSIModel(L.LightningModule):
    """This is the PyTorch lightning module that will be in charge of handling the CNN"""

    def __init__(self):
        """Initializes the model (Resnet50 with pretrained weights) and the loss 
        function (binary cross entropy)"""

        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
        self.loss_module = nn.BCEWithLogitsLoss(weight=torch.FloatTensor([1,1]))

    def set_class_weights(self, weights):
        """Apply weights to each class in the loss, to compensate for class imbalance in the dataset"""
        self.loss_module = nn.BCEWithLogitsLoss(weight=torch.FloatTensor(weights))

    def training_step(self, batch, batch_idx):
        """Standard PyTorch lightning method"""

        input_data, labels = batch
        preds = self.model(input_data).squeeze(dim=1)
        loss = self.loss_module(preds, labels)
        acc =  BinaryAccuracy().to(self.device)(preds, labels)
        self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=False, on_step=False, on_epoch=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        """Standard PyTorch lightning method"""

        input_data, labels = batch
        preds = self.model(input_data).squeeze(dim=1)
        loss = self.loss_module(preds, labels)
        acc =  BinaryAccuracy().to(self.device)(preds, labels)
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=False, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """Standard PyTorch lightning method"""

        input_data, labels = batch
        preds = self.model(input_data).squeeze(dim=1)
        acc = metric = BinaryAccuracy().to(self.device)(preds, labels)
        self.log("acc", acc)

    def forward(self, x):
        """Standard PyTorch lightning method"""
        return self.model(x).squeeze(dim=1)
        
    def configure_optimizers(self):
        """Standard PyTorch lightning method"""

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def train(data_dir, labels_filename, model):
    """Executes the training pipeline with a given dataset

    Args:
        - data_dir: path to the data directory containing the tiles and the csv file
        - labels_filename: csv file to use for annotations. Can vary depending on whether we use pseudo labels or not
        - model: the model to use. Should be an instance of WSIModel
    """

    # For reproducability
    torch.manual_seed(42) 

    # Build a dataset and split it in train/val/test subsets
    dataset = WSIDataset(data_dir, labels_filename)
    train_set, val_set, test_set = data.random_split(dataset, [0.7, 0.2, 0.1])

    train_loader = data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = data.DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4, persistent_workers=True)
    test_loader = data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

    # Define callbacks to retain the best model (minimal validation loss) and ensure the training stops if
    # the validation loss has not been improving for some time
    save_best_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor='val_loss',
        filename='best',
        save_top_k=1
    )
    early_stopping_callback = L.pytorch.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )

    callbacks = [
        save_best_callback,
        early_stopping_callback
    ]

    # Apply weights to the model loss function in order to compensate for class imbalance
    class_weights = [1.0, dataset.class_rep['NO_NUCLEUS']/dataset.class_rep['NUCLEI']]
    model.set_class_weights(class_weights)

    # Launch training
    logger = L.pytorch.loggers.tensorboard.TensorBoardLogger("tb_logs", name="WSIExperiments")
    trainer = L.Trainer(max_epochs=20, callbacks=callbacks, logger=logger)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test the model on the test set
    trainer.test(dataloaders=test_loader)

def generate_pseudo_labels(data_dir, labels_filename, model):
    """Generate automatic labels on unannotated data, based on the first training
    
    Args:
        - data_dir: path to the data directory containing the tiles and the csv file
        - labels_filename: csv file containing human annotations
        - model: the model to use. Should be an instance of WSIModel
        """

    model.eval()
    with torch.no_grad():

        # Find ids of tiles that were not annotated by humans
        tiles = list(Path(data_dir).glob('*.png'))
        pattern = r'(.*)tile_(?P<id>\d+).png'
        tiles_ids = [int(re.match(pattern, str(f)).groupdict()['id']) for f in tiles]
        annots = pd.read_csv(os.path.join(data_dir, labels_filename))
        missing_annots = [tid for tid in tiles_ids if tid not in annots['tile_id'].values]

        # Run model on each of those tiles, and add the resulting classification to the labels DataFrame
        transforms = torchvision.transforms.Compose(get_preprocessing_transforms())
        for tid in tqdm(missing_annots):
            f = os.path.join(data_dir, f'tile_{tid}.png')
            img = Image.open(f) 
            img = img.convert("RGB")
            img = transforms(img).to(model.device)
            res = model(img.unsqueeze(dim=0))
            label = "NUCLEI" if res[0][1]>res[0][0] else "NO_NUCLEUS"
            annots.loc[len(annots)] = [tid, label]

    # Save the pseudo labels for the second training
    annots.to_csv(os.path.join(data_dir, 'pseudo_labels.csv'), index=False)

if __name__ == '__main__':

    data_dir = "/data/data/dataset_nuclei_tiles"
    data_dir = "/Users/josselin/Downloads/primaa/data/dataset_nuclei_tiles"

    model = WSIModel()

    # First training with partially annotated dataset
    train(data_dir, 'labels.csv', model)

    # Generate pseudo labels
    print("\n\nComputing pseudo labels from first model\n\n")
    generate_pseudo_labels(data_dir, 'labels.csv', model)

    # Second training with automatic labels added to dataset
    print("\n\nRetraining with auto generated labels\n\n")    
    train(data_dir, 'pseudo_labels.csv', model)
    trainer.test(dataloaders=test_loader)