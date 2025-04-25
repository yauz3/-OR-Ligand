# µOR-Ligand: A Hybrid feature selection and Stacking Framework for µ-Opioid Receptor Ligand Activity Prediction

# Reference Implementation of Inter-POL algorithm
This readme file documents all of the required steps to run µOR-Ligand.

Note that the code was implemented and tested only on a Linux operating system.

# Data
The complete training and external validation data sets are freely downloadable at https://github.com/JanaShenLab/opioids or https://github.com/Myongin/muopioids.

## How to set up the environment
We have provided an Anaconda environment file for easy setup.
If you do not have Anaconda installed, you can get Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).
Create the `-OR-Ligand` environment using the following command:
```bash
conda env create -n -OR-Ligand -f environment.yml
conda activate -OR-Ligand
```

# In order to install the required packages
```bash
pip install -r requirements.txt
```

# Step by step files:

Step_1_prepare_ligand_features.py: prepare ligand features

Step_2_prepare_interaction_features.py: prepare molecular interaction features

Step_3_prepare_fingerprint_features.py: prepare molecular fingerprint features

Step_4_validate_performance.py: validate the µOR-Ligand

## License

This project is licensed for **academic and research purposes only**. For commercial usage, please connect with s.yavuz.ugurlu@gmail.com

## Acknowledgements
We thank the authors of Oh et. al. [1] for sharing their data.

## References
1. Oh, Myongin, et al. "Machine Learned Classification of Ligand Intrinsic Activities at Human μ-Opioid Receptor." ACS Chemical Neuroscience 15.15 (2024): 2842-2852.
