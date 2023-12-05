from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from typing import Type


def objective(
    pipeline: Type[LightningModule],
    params: dict,
    train: DataLoader,
    val: DataLoader,
    n: int,
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
    pipeline.set_params(**params)
    pipeline.fit(X_train, y_train)
    results = trainer.test(model, dataloaders=val_loader)
    return results[-1][metric]


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
        pipeline: Pipeline,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        sample_size: Optional[int] = None,
        metric: str = "roc_auc",
        stratify: bool = True,
        n_splits: int = 5,
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
        self.params = config
        self.pipeline = pipeline
        self.X_train = X_train
        self.y_train = y_train
        self.sample_size = sample_size
        self.metric = metric
        self.scores = np.array([])

        if stratify:
            self.splitter = StratifiedKFold(n_splits)
        else:
            self.splitter = KFold(n_splits)

        self.fold_indices = []
        if not sample_size:
            for train_index, test_index in self.splitter.split(
                X_train,
                y_train,
            ):
                self.fold_indices.append((train_index, test_index))
        else:
            for train_index, test_index in self.splitter.split(
                X_train.sample(sample_size, seed=1),
                y_train.sample(
                    sample_size,
                    seed=1,
                ),
            ):
                self.fold_indices.append((train_index, test_index))

    def step(self):
        """
        Perform a training step.

        Returns:
        dict: A dictionary containing the score for the current step.
        """
        try:
            score = objective(
                self.pipeline,
                self.params,
                self.X_train[self.fold_indices[self.x][0]],
                self.y_train[self.fold_indices[self.x][0]],
                self.X_train[self.fold_indices[self.x][1]],
                self.y_train[self.fold_indices[self.x][1]],
                self.x,
                self.metric,
            )
            self.scores = np.append(self.scores, score)
            self.x += 1
        except:
            print(f"cross val {self.x} False")
        return {"score": self.scores.mean()}
