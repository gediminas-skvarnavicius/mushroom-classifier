from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning.loggers import Logger
from typing import Type, Optional, List
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ray import tune


def config_models(
    config,
    model: Type[LightningModule],
    dm: Type[LightningDataModule],
):
    model = model(
        num_classes=config["num_classes"],
        learning_rate=config["learning_rate"],
        architecture=config["architecture"],
        optimizer=config["optimizer"],
        l2=config["l2"],
    )
    dm = dm(batch_size=config["batch_size"], img_size=config["img_size"])
    return model, dm


class TrainableCV(tune.Trainable):
    """
    A custom Ray Tune trainable class for hyperparameter tuning.

    This class is used to configure and execute hyperparameter
    tuning experiments using Ray Tune. It sets up the necessary
    parameters and data for each trial, and performs steps to
    evaluate the hyperparameter configurations.

    Attributes:
    - config (dict): A dictionary of hyperparameters for the pipeline.
    - pipeline: The machine learning pipeline to be configured and evaluated.
    - X_train: Training data features.
    - y_train: Training data labels.
    - sample_size (Union[int, str]): The sample size for data splitting.
    - metric (str): The metric used.
    - stratify (bool): Whether to stratify data splitting.

    Methods:
    - setup(config, pipeline, X_train, y_train, X_val, y_val, sample_size,
    metric, stratify):
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
        callbacks: Optional[List[EarlyStopping]] = None,
    ):
        """
        Set up the trainable object with hyperparameters and data.

        Args:
        config (dict): A dictionary of hyperparameters.
        pipeline: The machine learning pipeline.
        X_train: Training data features.
        y_train: Training data labels.
        sample_size (Union[int, str], optional): The sample size for data
        splitting.
        n_splits: The number of splits for cross-validation.

        metric (str, optional): The metric used for scoring.

        stratify (bool, optional): Whether to stratify data splitting.
        Default is True.
        """

        self.x = 0
        self.config = config
        self.model = model(
            num_classes=config["num_classes"],
            learning_rate=config["learning_rate"],
            architecture=config["architecture"],
            optimizer=config["optimizer"],
            l2=config["l2"],
        )
        self.dm = dm(
            batch_size=config["batch_size"],
            img_size=config["img_size"],
        )
        self.metric = metric
        self.scores = np.array([])
        self.trainer = Trainer(
            max_epochs=config["max_epochs"],
            accelerator="gpu",
            callbacks=callbacks,
            logger=logger,
        )

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
        - pipeline (Pipeline): The machine learning pipeline to be evaluated.
        - params (dict): Hyperparameter configuration for the pipeline.
        - X_train (pl.DataFrame): Training data features as a Polars DataFrame.
        - y_train (pl.Series): Training data labels as a Polars Series.
        - X_val (pl.DataFrame): Validation data features as a Polars DataFrame.
        - y_val (pl.Series): Validation data labels as a Polars Series.
        - n (int): The current iteration number.
        - metric (str, optional): The metric for scoring. Supported metrics:
        "roc_auc", "f1", "rmse".

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
