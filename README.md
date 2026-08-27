
## Zinc Data Setup
To set up the data necessary for MoSE experiments on ZINC, unzip the `hombasis-gt/hombasis-bench/data/zinc-data.zip` file into the `hombasis-gt/hombasis-bench/data` directory. 

## QM9 Data Setup
To set up the data necessary for MoSE experiments on QM9, unzip the file `hombasis-gt/qm9/data/QM9/v5_homcounts.zip`, and move the resulting files (`test_homcounts.json`, `train_homcounts.json`, `valid_homcounts.json`) into the `hombasis-gt/qm9/data/QM9` directory. Then, run the python script `hombasis-gt/qm9/data_GraphGym_QM9/save_qm9_hc.py` in order to process the count-enhanced QM9 dataset (will be saved as `datasets/QM9-GraphHC/processed/joined.pt`). It may take a few minutes for `save_qm9_hc.py` to run.

## PCQM4Mv2-subset Data Setup
Get the MoSE enhanced PCQM4Mv2-subset dataset file [here](https://drive.google.com/drive/folders/1NjdueZ3D9sJWX2n4f3P5loYgRxmYF2LN?usp=sharing). Set up the dataset by unzipping the downloaded file and moving all resulting json files into the `hombasis-gt/pcqm/data` directory.

## CIFAR10 & MNIST Data Setup
Get the MoSE enhanced CIFAR10 and MNIST dataset files [here](https://drive.google.com/drive/folders/1NjdueZ3D9sJWX2n4f3P5loYgRxmYF2LN?usp=sharing). Set up the datasets by unzipping the downloaded files into the `hombasis-gt/image-datasets/data` directory.

## Synth Data Setup
To set up our synthetic dataset, run the script `hombasis-gt/synth/save_synth_dataset.py` (this will save the homomorphism count enhanced datasets to `datasets/SYNTH-All5/processed` and `datasets/SYNTH-Spasm/processed`). 

## Peptides Data Setup
To set up the MoSE enhanced Peptides Functional and Structural dataset, simply unzip the `hombasis-gt/peptides/data/raw_data.zip` file and move the resulting json files (`peptides_c78.json` and `peptides_v5c6.json`) into the `hombasis-gt/peptides/data` directory.

## Running Experiments
To run an experiment, set up a `configuration.yaml` file containing the model hyperparameters and experimental setup such as those given in the `GraphGPS/configs/` directory. Then, run:

```bash
python GraphGPS/main.py --cfg "/path_to/configuration.yaml" --repeat 1 wandb.use True
```

For example, replicate the best result for ZINC GPS+Spasm by running:

```bash
python GraphGPS/main.py --cfg "./GraphGPS/configs/ZINC/With_Edge_Features/GPSe/+spasm.yaml" --repeat 1 wandb.use True seed 0
python GraphGPS/main.py --cfg "./GraphGPS/configs/ZINC/With_Edge_Features/GPSe/+spasm.yaml" --repeat 1 wandb.use True seed 14
python GraphGPS/main.py --cfg "./GraphGPS/configs/ZINC/With_Edge_Features/GPSe/+spasm.yaml" --repeat 1 wandb.use True seed 48
python GraphGPS/main.py --cfg "./GraphGPS/configs/ZINC/With_Edge_Features/GPSe/+spasm.yaml" --repeat 1 wandb.use True seed 96
```

To run experiments with GRIT, use the `GRIT/main.py` script instead of `GraphGPS/main.py`, and use config files given in `GRIT/configs/`.
