
# Mushroom Classifier
This project involves the creation of a computer vision model for predicting mushroom genera, leveraging transfer learning techniques.
## Contents

The main notebook: [mushroom_analysis.ipynb](mushroom_analysis.ipynb).

[Model Files](model_files) contain functions and classes used in the project:

* [model.py](model_files/model.py) contains model related classes MushroomClassifier, MushroomDataModule and MushroomTuner
* [tuning.py](model_files/tuning.py) contains setup for configuration tuning using Ray-Tune.
* [explainer.py](model_files/explainer.py) contains a function for model visual explanation.

The tensorboard training logs of the final model can be fined in the [clean_logs](clean_logs) folder.