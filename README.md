
# VLMath

This repository contains the code, datasets, and results for the **VLMath** paper.  
It includes training scripts, evaluation results, and analyses based on the experiments described in the manuscript and supplementary materials.

---

## Contents

- `Datasets/` — Data used for training and evaluation  
- `Library/` — Core scripts for training and evaluation  
- `results/` — Benchmark results, training logs, and analysis outputs  

---

## Training

To reproduce training:

```bash
conda activate phi_env
python main.py train --model-name phi --run-name <experiment_name>
