from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    squeezenet1_0,
    SqueezeNet1_0_Weights,
    googlenet,
    GoogLeNet_Weights,
)  # type:ignore
from torch.nn.functional import cross_entropy
from torch.nn import Linear
from torch.optim import Adam, SGD
from pytorch_lightning import LightningModule, LightningDataModule
import torch
from PIL import Image
from torchvision import datasets, transforms
import polars as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import numpy as np
from torcheval.metrics.functional import multiclass_f1_score


def image_val(path, img_size=225):
    try:
        image = Image.open(path)
        image = image.resize((img_size, img_size))
        return True
    except:
        return False


class MushroomDataModule(LightningDataModule):
    def __init__(self, batch_size=128, img_size=224):
        super().__init__()
        self.img_size = img_size
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def setup(self, stage=None):
        dataset = datasets.ImageFolder(
            root="/home/gediminas/Documents/turing_projects/module4_s1/gskvar-DL.1.5/Mushrooms",
            transform=self.transform,
            is_valid_file=lambda path: image_val(
                path=path,
                img_size=self.img_size,
            ),
        )
        path_df = pl.DataFrame(dataset.samples, schema=["img", "class"])
        path_df = path_df.with_columns(pl.Series(np.arange(len(path_df))).alias("id"))
        # Getting a list of data point indexes for a stratified split
        train_idx, valid_test_idx = train_test_split(
            path_df, stratify=path_df["class"], test_size=0.4, random_state=1
        )
        train_idx = train_idx["id"].to_list()
        valid_idx, test_idx = train_test_split(
            valid_test_idx,
            stratify=valid_test_idx["class"],
            test_size=0.5,
            random_state=1,
        )
        valid_idx = valid_idx["id"].to_list()
        test_idx = test_idx["id"].to_list()
        # Subsets based on index
        self.train_set = Subset(dataset, train_idx)
        self.valid_set = Subset(dataset, valid_idx)
        self.test_set = Subset(dataset, test_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=5,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            num_workers=5,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=5,
        )


class MushroomClassifier(LightningModule):
    def __init__(
        self,
        num_classes=9,
        learning_rate=1e-3,
        architecture="resnet18",
        optimizer="adam",
        l2=0,
    ):
        super().__init__()

        # Load pre-trained architecture
        models = {
            "resnet18": resnet18(weights=ResNet18_Weights.DEFAULT),
            "squeezenet": squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT),
            "google": googlenet(weights=GoogLeNet_Weights.DEFAULT),
        }
        self.model = models[architecture]
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.l2 = l2
        # Modify the classifier to match the number of classes in your dataset
        if architecture == "squeezenet":
            in_features = self.model.classifier[1].in_channels
        else:
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
        optimizers = {
            "adam": Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.l2,
            ),
            "sgd": SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.l2,
            ),
        }
        optimizer = optimizers[self.optimizer]
        return optimizer

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = cross_entropy(outputs, labels)
        f1 = multiclass_f1_score(
            labels,
            torch.argmax(outputs, dim=1),
            average="weighted",
            num_classes=9,
        )
        # Log any additional metrics you are interested in
        acc = (torch.argmax(outputs, dim=1) == labels).float().mean()

        self.log("test_loss", loss)
        self.log("test_accuracy", acc, prog_bar=True)
        self.log("test_weighted_f1_score", f1)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, labels = batch
        return self(inputs)
