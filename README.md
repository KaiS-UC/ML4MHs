# Machine Learning for Metal Hydrides (ML4MHs)

This repository contains a collection of python scripts for building and testing machine learning models of metal hydrides. The details of our methodology can be found in the associated publication [a link will appear here once published]. There is no active development of this software, but if you have suggestions for improvement, feel free to raise an issue.

## Usage
To run the scripts, you will need a few python packages that are listed in the requirements.txt file. You can install them with pip, using
```pip install -r requirements.txt```
We highly recommend installing packages in a separate environment to keep the rest of your python installation unaffected by the changes.

Once all required packages are installed, you should be able to reproduce the results from the paper by running the scripts in the following sequence:
```
python select_features.py
python train_models.py
python generate_plots.py
```

If you want to apply the same workflow to a different dataset, simply put your dataset of choice in a suitable format into the directory `training_data`, and adapt the corresponding path in `select_features.py` (search for the variable datafile).

If you want to update the hydrogenation data parsed from the MaterialsProject, you will have to provide your own api key in the corresponding variable in `select_features.py` (search for the variable MP_API_KEY).

## How to cite
If you make use of our code in any of your publications, please cite our corresponding paper. A link to the paper will appear in this section once it is published. Please check the latest version of the repository. If this version is still the latest, you may cite this software as:

K. Sellschopp and S. Faouri (2025), *Machine Learning for Metal Hydrides (ML4MHs)* version 1.0 [Computer software], https://github.com/KaiS-UC/ML4MHs/releases/tag/v1.0