from torchvision.models import resnet18, ResNet18_Weights  # type:ignore
from torch.nn.functional import cross_entropy
from torch.nn import Linear
from torch.optim import Adam
from pytorch_lightning import LightningModule, LightningDataModule
import torch
from PIL import Image
from torchvision import datasets, transforms
import polars as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import numpy as np


def image_val(path):
    try:
        image = Image.open(path)
        image = image.resize((225, 225))
        return True
    except:
        return False


class MushroomDataModule(LightningDataModule):
    def __init__(self, batch_size=128):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        def setup(self, stage=None):
            dataset = datasets.ImageFolder(
                root="Mushrooms",
                transform=self.transform,
                is_valid_file=image_val,
            )
            path_df = pl.DataFrame(dataset.samples, schema=["img", "class"])
            path_df = path_df.with_columns(
                pl.Series(np.arange(len(path_df))).alias("id")
            )
            # Getting a list of data point indexes for a stratified split
            path_df = pl.DataFrame(dataset.samples, schema=["img", "class"])
            train_idx, valid_test_idx = train_test_split(
                path_df, stratify=path_df["class"], test_size=0.4
            )
            train_idx = train_idx["id"].to_list()
            valid_idx, test_idx = train_test_split(
                valid_test_idx, stratify=valid_test_idx["class"], test_size=0.5
            )
            valid_idx = valid_idx["id"].to_list()
            test_idx = test_idx["id"].to_list()
            # Subsets based on index
            self.train_set = Subset(dataset, train_idx)
            self.valid_set = Subset(dataset, valid_idx)
            self.test_set = Subset(dataset, test_idx)

        def train_dataloader(self):
            return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=5)

        def val_dataloader(self):
            return DataLoader(
                self.valid_set,
                batch_size=self.batch_size,
                num_workers=5,
            )

        def test_dataloader(self):
            return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=5)


class MushroomClassifier(LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super(MushroomClassifier, self).__init__()

        # Load pre-trained ResNet-18
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Modify the classifier to match the number of classes in your dataset
        in_features = self.model.fc.in_features
        self.model.fc = Linear(in_features, num_classes)
        # Set other hyperparameters
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = cross_entropy(outputs, labels)
        self.logger.experiment.add_scalars(
            "loss",
            {"train": loss},
            self.global_step,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = cross_entropy(outputs, labels)
        self.logger.experiment.add_scalars(
            "loss",
            {"val": loss},
            self.global_step,
        )
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = cross_entropy(outputs, labels)

        # Log any additional metrics you are interested in
        acc = (torch.argmax(outputs, dim=1) == labels).float().mean()
        self.log("test_loss", loss)
        self.log("test_accuracy", acc, prog_bar=True)

        return loss
