from IPython.display import clear_output
from PIL import Image  # type:ignore
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from lime import lime_image  # type:ignore
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from pytorch_lightning import LightningModule, Trainer
from torchvision.transforms.transforms import Compose  # type:ignore


def explain_model(
    path: str,
    trainer: Trainer,
    model: LightningModule,
    transform: Compose,
    n_samples: int = 1000,
    return_img=False,
) -> None:
    """
    Explain a machine learning model's predictions on an image using LIME.

    Parameters:
    - path (str): The path to the image file.
    - trainer (pytorch_lightning.Trainer): The trainer object used to make
    predictions.
    - model (pytorch_lightning.LightningModule): The machine learning model
    to be explained.
    - transform (Compose): A function to transform
    the input image.
    - n_samples (int, optional): The number of samples used by LIME for
    explanation. Default is 1000.

    Returns:
    None
    """
    # Load the image
    image = Image.open(path)

    # Function to predict the image
    def predict_image(image):
        arrays = [
            transform(Image.fromarray(i.squeeze(0)))
            for i in np.split(image, image.shape[0], axis=0)
        ]
        dl = DataLoader(
            TensorDataset(
                torch.stack(arrays),
                torch.tensor(np.zeros(10, dtype=int)),
            ),
            10,
        )
        predictions = trainer.predict(model, dl)
        clear_output()
        return predictions[0]

    # LimeImageExplainer
    explainer = lime_image.LimeImageExplainer()

    # Explain the instance
    explanation = explainer.explain_instance(
        np.array(image), predict_image, labels=(0), num_samples=n_samples
    )

    # Get the image and mask
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False,
    )

    # Mark boundaries on the image
    img_boundary = mark_boundaries(temp / 255.0, mask)

    if return_img:
        return img_boundary
    else:
        plt.imshow(img_boundary)
        plt.show()
