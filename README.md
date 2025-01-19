# Dataset

Before initial use, extract the dataset.

```
tar -xf dataset/pre_submit_dataset.tar.xz -C dataset
tar -xf dataset/post_submit_dataset.tar.xz -C dataset
```

The dataset for each pipeline can now be found in `datasets/` as `.csv` files.

# Replication

## Dataset Preprocessing

`Dataset.py` loads the dataset csv files and performs preprocessing such as encoding of the test target names.
The preprocessed datasets with be stored in the `dataset` folder as `.pkl` files

## Model Execution

The regression test optimization models for pre- and post-submit are split into two different files.
To execute it, run `PostSubmitPipelineAgent.py` or `PreSubmitPipelineAgent.py`. This will run the model and save all test suite prioritization and selection results as a pandas dataframe in a `.pkl` file in `results`.
