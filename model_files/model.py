from torchvision.models import resnet18, ResNet18_Weights  # type:ignore
from torch.nn.functional import cross_entropy
from torch.nn import Linear
from torch.optim import Adam
from pytorch_lightning import LightningModule
import torch


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
