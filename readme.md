This repository contains code for my thesis used for evaluation of a face recognition pipeline on the LFW dataset.

## Data

To download the dataset run `data/download_lfw.sh`

For creating a subset of the dataset, run `data/subset_lfw_relative_path.sh`

## Evaluation

To re-run the LFW evaluation you can run either `FR_pipeline_evaluation_phase1.ipynb` or `FR_pipeline_evaluation_phase2.ipynb`, depending on which pipeline you want to evaluate. Before running this notebook, make sure you have downloaded the data in the data folder and specified this path in the beginning of the notebook. Actual evaluation of the LFW dataset in these notebook can be found under the final "Evalution" header.
