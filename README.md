
# VLMath

## Overview

VLMath is a multimodal tutoring framework designed to generate teacher–student dialogues that combine visual reasoning with pedagogical intent.
The model learns to not only solve problems correctly but also to teach, question, and guide reasoning in a human-like, instruction-aware manner.
Through targeted fine-tuning on synthetic dialogues, VLMath demonstrates how structured pedagogical conditioning can align large vision–language models with effective tutoring behavior.
 
## Pedagogical Prompting Strategies

We experiment with three pedagogical prompting strategies to explore how different instructional paradigms affect multimodal tutoring:
Mistake Correction – The teacher evaluates the student’s reasoning, identifies specific errors, and explains how to correct them step by step.
Scaffolding – The teacher offers gradual guidance and reflective hints, helping the student reason through the problem without revealing the answer.
Socratic Questioning – The teacher engages the student through focused, open-ended questions that promote self-explanation and reflective discovery.
While all three strategies were tested under identical conditions, the Socratic Questioning dataset was used to train the final VLMath model.
This variant produces concise, inquiry-based dialogues that balance reasoning accuracy, conversational fluency, and pedagogical alignment.

---



## Contents

- `Config/` - Accelerate Config used for training
- `Datasets/` - Data used for training and evaluation
- `Library/` - Core scripts for training
- `DatasetGeneration/` - Scripts and prompts used for creating datasets
- `results/` - Benchmark results and token analysis outputs  

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
