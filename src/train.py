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
        TODO: documentation
        """

        super().__init__()

        tiles_files = list(Path(data_dir).glob('*.png'))

        self.data_points = []
        annots = pd.read_csv(os.path.join(data_dir, labels_filename))
        for index, row in annots.iterrows():
            filepath = os.path.join(data_dir, f'tile_{row["tile_id"]}.png')
            if not os.path.exists(filepath):
                print(f"Did not find {filepath}")
                continue
            nuclei = 0 if row['label'] == "NO_NUCLEUS" else 1
            self.data_points.append((filepath, torch.FloatTensor([1-nuclei, nuclei])))

        self.class_rep = annots['label'].value_counts()

        transforms = get_preprocessing_transforms()
        transforms.insert(0, torchvision.transforms.RandomRotation(degrees=180))
        self.transforms = torchvision.transforms.Compose(transforms)


    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        path, label = self.data_points[idx]

        with open(path, "rb") as f: 
            img = Image.open(f) 
            img = img.convert("RGB")
            img = self.transforms(img)

        return img, label

class WSIModel(L.LightningModule):
    """TODO: doc"""

    def __init__(self):
        """TODO"""

        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
        self.loss_module = nn.BCEWithLogitsLoss(weight=torch.FloatTensor([1,1]))

    def set_class_weights(self, weights):
        """TODO"""
        self.loss_module = nn.BCEWithLogitsLoss(weight=torch.FloatTensor(weights))

    def training_step(self, batch, batch_idx):
        """TODO"""
        input_data, labels = batch
        preds = self.model(input_data).squeeze(dim=1)
        loss = self.loss_module(preds, labels)
        acc =  BinaryAccuracy().to(self.device)(preds, labels)
        self.log("train_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=False, on_step=False, on_epoch=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        """TODO"""
        input_data, labels = batch
        preds = self.model(input_data).squeeze(dim=1)
        loss = self.loss_module(preds, labels)
        acc =  BinaryAccuracy().to(self.device)(preds, labels)
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=False, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        """TODO"""

        input_data, labels = batch
        preds = self.model(input_data).squeeze(dim=1)
        acc = metric = BinaryAccuracy().to(self.device)(preds, labels)
        self.log("acc", acc)

    def forward(self, x):
        """TODO"""

        return self.model(x).squeeze(dim=1)
        
    def configure_optimizers(self):
        """TODO"""

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def train(data_dir, labels_filename, trainer=None):
    """TODO"""

    torch.manual_seed(42) # For reproducability

    dataset = WSIDataset(data_dir, labels_filename)
    train_set, val_set, test_set = data.random_split(dataset, [0.7, 0.2, 0.1])

    train_loader = data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = data.DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4, persistent_workers=True)
    test_loader = data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=4)

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

    class_weights = [1.0, dataset.class_rep['NO_NUCLEUS']/dataset.class_rep['NUCLEI']]

    logger = L.pytorch.loggers.tensorboard.TensorBoardLogger("tb_logs", name="WSIExperiments")
    trainer = L.Trainer(max_epochs=20, callbacks=callbacks, logger=logger)
    model.set_class_weights(class_weights)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(dataloaders=test_loader)

def generate_pseudo_labels(data_dir, labels_filename, model):
    """TODO"""

    model.eval()
    with torch.no_grad():
        tiles = list(Path(data_dir).glob('*.png'))
        pattern = r'(.*)tile_(?P<id>\d+).png'
        tiles_ids = [int(re.match(pattern, str(f)).groupdict()['id']) for f in tiles]
        annots = pd.read_csv(os.path.join(data_dir, labels_filename))
        missing_annots = [tid for tid in tiles_ids if tid not in annots['tile_id'].values]

        transforms = torchvision.transforms.Compose(get_preprocessing_transforms())
        for tid in tqdm(missing_annots):
            f = os.path.join(data_dir, f'tile_{tid}.png')
            img = Image.open(f) 
            img = img.convert("RGB")
            img = transforms(img).to(model.device)
            res = model(img.unsqueeze(dim=0))
            label = "NUCLEI" if res[0][1]>res[0][0] else "NO_NUCLEUS"
            annots.loc[len(annots)] = [tid, label]

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