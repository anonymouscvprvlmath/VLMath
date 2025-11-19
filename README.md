
# VLMath

This repository contains the code, datasets, and results for the **VLMath** paper.  
It includes training scripts, evaluation results, and analyses based on the experiments described in the manuscript and supplementary materials.

---



## Contents

- `Datasets/` — Data used for training and evaluation  
- `Library/` — Core scripts for training and evaluation  
- `results/` — Benchmark results, training logs, and analysis outputs  

---

## Environment

To setup the environment 
```bash
conda env create -f environment.yml
```

## Training

To reproduce training:

```bash
conda activate VLMath_env
accelerate launch --config_file ./Config/phi_ds_config.yaml main.py train --process-dataset True --model-name phi --run-name <experiment_name>
