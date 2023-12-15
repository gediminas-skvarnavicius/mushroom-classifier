from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from typing import Type, Optional, Tuple
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ray import tune
from .model import MushroomClassifier, MushroomDataModule, MushroomTuner
from pytorch_lightning.loggers import Logger


def config_trainer_model_dm(
    config: dict,
    model: Type[MushroomDataModule],
    dm: Type[MushroomClassifier],
    logger: Optional[Logger] = None,
) -> Tuple[Trainer, MushroomClassifier, MushroomDataModule]:
    """
    Configure and initialize a model, data module, and trainer based on the
    provided configuration.

    Parameters:
    - config (dict): A dictionary containing configuration parameters.
    - model (Type[MushroomDataModule]): The class type for the model.
    - dm (Type[MushroomClassifier]): The class type for the data module.
    - logger: Optional logger for logging training information.

    Returns:
    Tuple[Trainer, MushroomClassifier, MushroomDataModule]: A tuple containing
    the initialized trainer, model, and data module.
    """
    model_instance = model(
        num_classes=config["num_classes"],
        learning_rate=config["learning_rate"],
        architecture=config["architecture"],
        optimizer=config["optimizer"],
        l2=config["l2"],
    )
    dm_instance = dm(
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        base_dir=config["base_dir"],
    )

    callbacks = []
    # early stopping
    if config["early_stopping_params"] != None:
        callbacks.append(EarlyStopping(**config["early_stopping_params"]))
    # Fine Tuning
    if config["fine_tuning_params"] != None:
        callbacks.append(MushroomTuner(**config["fine_tuning_params"]))
    trainer_instance = Trainer(
        max_epochs=config["max_epochs"],
        accelerator="gpu",
        callbacks=callbacks,
        logger=logger,
    )
    return trainer_instance, model_instance, dm_instance


class TrainableP2L(tune.Trainable):
    """
    A custom Ray Tune trainable class for hyperparameter tuning of
    PyTorch Lightning models.

    This class is used to configure and execute hyperparameter
    tuning experiments using Ray Tune. It sets up the necessary
    parameters and data for each trial, and performs steps to
    evaluate the hyperparameter configurations.

    Attributes:
    - config (dict): A dictionary of hyperparameters for the model.
    - model: The PyTorch Lightning model to be configured and evaluated.
    - dm: The PyTorch Lightning DataModule for handling data.
    - metric (str): The metric used for evaluation.
    - logger (Optional[Logger]): An optional logger for experiment logging.

    Methods:
    - setup(config, model, dm, metric, logger, callbacks):
        Set up the trainable object with hyperparameters and data.

    - step():
        Perform a training step and return the score.

    """

    def setup(
        self,
        config: dict,
        model: Type[LightningModule],
        dm: Type[LightningDataModule],
        metric: str = "val_loss",
        logger: Optional[Logger] = None,
    ):
        """
        Set up the trainable object with hyperparameters and data.

        Args:
        config (dict): A dictionary of hyperparameters.
        model: The PyTorch Lightning model.
        dm: The PyTorch Lightning DataModule.
        metric (str, optional): The metric used for scoring.
        logger (Optional[Logger], optional): An optional logger for
        experiment logging.
        """

        self.x = 0
        self.config = config
        self.trainer, self.model, self.dm = config_trainer_model_dm(
            config, model, dm, logger=logger
        )
        self.metric = metric
        self.scores = np.array([])

    @staticmethod
    def objective(
        trainer: Trainer,
        model: Type[LightningModule],
        dm: Type[LightningDataModule],
        metric: str = "val_loss",
    ) -> float:
        """
        Objective function for hyperparameter tuning.

        Parameters:
        - trainer (Trainer): The Lightning Trainer for training the model.
        - model (Type[LightningModule]): The LightningModule representing
        the machine learning model.
        - dm (Type[LightningDataModule]): The LightningDataModule
        containing the data for training and validation.
        - metric (str, optional): Name of the metric.

        Returns:
        - float: The score based on the specified metric.
        """

        trainer.fit(model, datamodule=dm)
        return trainer.callback_metrics[metric].item()

    def step(self):
        """
        Perform a training step.

        Returns:
        dict: A dictionary containing the score for the current step.
        """
        score = self.objective(
            model=self.model,
            trainer=self.trainer,
            dm=self.dm,
            metric=self.metric,
        )
        self.scores = np.append(self.scores, score)
        return {"score": self.scores.mean()}
