from torchvision.models import (  # type: ignore
    resnet18,
    ResNet18_Weights,
    squeezenet1_0,
    SqueezeNet1_0_Weights,
    googlenet,
    GoogLeNet_Weights,
)  # type: ignore
from torch.nn.functional import cross_entropy
from torch.nn import Linear, Sequential
from torch.optim import Adam, SGD
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import BaseFinetuning
import torch
from PIL import Image  # type: ignore
from torchvision import datasets, transforms  # type: ignore
import polars as pl
from sklearn.model_selection import train_test_split  # type: ignore
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import numpy as np
from torcheval.metrics.functional import multiclass_f1_score
import os
from typing import Optional, Tuple


def image_val(path: str, img_size: int = 225) -> bool:
    """
    Validates an image file.

    Args:
        path (str): The path to the image file.
        img_size (int, optional): The size of the image. Default is 225.
    Returns:
        bool: True if the image is successfully opened and resized,
        False otherwise.
    """
    try:
        image = Image.open(path)
        image = image.resize((img_size, img_size))

        return True
    except:
        return False


class MushroomDataModule(LightningDataModule):
    """
    LightningDataModule subclass for handling Mushroom dataset.

    Args:
        batch_size (int): The batch size for data loaders. Default is 128.
        img_size (int): The size of the input images. Default is 224.
    """

    def __init__(
        self, batch_size: int = 128, img_size: int = 224, base_dir: str = None
    ):
        """
        MushroomDataModule constructor.

        Args:
            batch_size (int): Batch size for DataLoader.
            img_size (int): Size of the input images.
        """
        super().__init__()
        self.img_size = img_size
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        if base_dir == None:
            self.BASE_DIR = os.getcwd()
        else:
            self.BASE_DIR = base_dir

    def setup(self, stage: Optional[str] = None):
        """
        Set up the Mushroom dataset.

        Args:
            stage (str, optional): The stage of setup. Default is None.
        """
        dataset = datasets.ImageFolder(
            root=f"{self.BASE_DIR}/Mushrooms",
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

    def train_dataloader(self) -> DataLoader:
        """
        Returns a data loader for training set.

        Returns:
            torch.utils.data.DataLoader: Data loader for training set.
        """
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=5,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns a data loader for validation set.

        Returns:
            torch.utils.data.DataLoader: Data loader for validation set.
        """
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            num_workers=5,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns a data loader for test set.

        Returns:
            torch.utils.data.DataLoader: Data loader for test set.
        """
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=5,
        )


class MushroomClassifier(LightningModule):
    def __init__(
        self,
        num_classes: int = 9,
        learning_rate: float = 1e-3,
        architecture: str = "resnet18",
        optimizer: str = "adam",
        l2: float = 0,
    ) -> None:
        """
        Initializes the MushroomClassifier.

        Args:
            num_classes (int): The number of classes in the dataset.
                Default is 9.
            learning_rate (float): The learning rate for the optimizer.
                Default is 1e-3.
            architecture (str): The architecture of the model.
                Default is "resnet18".
            optimizer (str): The optimizer to use. Default is "adam".
            l2 (float): The L2 regularization strength. Default is 0.
        """
        super().__init__()

        # Load pre-trained architecture
        models = {
            "resnet18": resnet18(weights=ResNet18_Weights.DEFAULT),
            "squeezenet": squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT),
            "google": googlenet(weights=GoogLeNet_Weights.DEFAULT),
        }
        backbone = models[architecture]
        self.num_classes = num_classes
        # backbone number of features
        if architecture == "squeezenet":
            in_features = backbone.classifier[1].in_channels
        else:
            in_features = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = Sequential(*layers)
        self.model = Linear(in_features, num_classes)
        # Set other hyperparameters and optimizer
        self.learning_rate = learning_rate
        self.l2 = l2
        self.optimizer = optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step of the model.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss tensor.
        """
        inputs, labels = batch
        outputs = self(inputs)
        loss = cross_entropy(outputs, labels)
        if self.logger:
            self.logger.experiment.add_scalars(
                "loss",
                {"train": loss},
                self.global_step,
            )
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step of the model.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss tensor.
        """
        inputs, labels = batch
        outputs = self(inputs)
        loss = cross_entropy(outputs, labels)
        if self.logger:
            self.logger.experiment.add_scalars(
                "loss",
                {"val": loss},
                self.global_step,
            )
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        optimizers = {
            "adam": Adam,
            "sgd": SGD
            # "adam": Adam(
            #     self.parameters(),
            #     lr=self.learning_rate,
            #     weight_decay=self.l2,
            # ),
            # "sgd": SGD(
            #     self.parameters(),
            #     lr=self.learning_rate,
            #     weight_decay=self.l2,
            # ),
        }
        optimizer = optimizers[self.optimizer]
        return optimizer(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            weight_decay=self.l2,
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Test step of the model.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The input batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss tensor.
        """
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

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """
        Predict step of the model.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The input batch.
            batch_idx (int): The index of the batch.
            dataloader_idx (int): The index of the dataloader. Default is 0.

        Returns:
            torch.Tensor: The output tensor.
        """
        inputs, labels = batch
        return self(inputs)


class MushroomTuner(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=5, tuning_lr=1e-5):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self.tuning_lr = tuning_lr

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.feature_extractor, train_bn=False)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor,
                optimizer=optimizer,
                lr=self.tuning_lr,
                train_bn=True,
            )
