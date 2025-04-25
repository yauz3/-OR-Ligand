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

# In order to install requirement packages
```bash
pip install -r requirements.txt
```

# Step by step files:

1_Generate_features.py: prepare features

2_Inter-hammet_train_and_evaluate.py: train and test the model


## License

This project is licensed for **academic and research purposes only**. For commercial usage, please connect with s.yavuz.ugurlu@gmail.com
