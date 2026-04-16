# AlgoOverride: Measuring LLM Inhibitory Control via Corrupted Cellular Automata

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-Hackathon_Submission-blue.svg)](https://www.kaggle.com/competitions/kaggle-measuring-agi/overview)

This repository contains the dataset generator, evaluation scripts, and benchmark data for **AlgoOverride**, a novel AI evaluation framework designed to measure *inhibitory control* in Large Language Models (LLMs).

## 🧠 The Core Concept
Current LLMs easily ace standard programming evaluations, but this often masks a deficit in executive functioning. When a habitual response pattern is rendered incorrect by a novel context, models struggle to override their pre-training. 

**AlgoOverride** operationalizes this by mutating the rules of Conway's Game of Life—one of the most heavily memorized deterministic systems in LLM training data. Models are presented with a grid and a mutated rule (e.g., *"Birth requires exactly 4 live neighbors, not 3"*). To succeed, the model must actively suppress its trained prior and simulate the new state machine.

We introduce the **Rule Override Score (ROS)**, a strict, exact-string-match metric that measures a model's cognitive flexibility and ability to abandon memorized heuristics.

<img width="8601" height="5734" alt="Corrupted Conway&#39;s game of life mutations" src="https://github.com/user-attachments/assets/fcc04264-f763-4882-8f84-552105f8e4db" />

<img width="1024" height="1536" alt="Corrupted Game of Life mutations" src="https://github.com/user-attachments/assets/e93a69b1-3550-46dd-bec0-3de3c1fb379c" />




## 📂 Repository Structure

* `dataset_generator.py` - The Python script used to procedurally generate the 600 synthetic benchmark tasks, including the 150 "Pattern Trap" adversarial tasks.
* `AlgoOverride_Benchmark.ipynb` - The Kaggle-compatible Jupyter Notebook containing the evaluation pipeline, custom regex extraction parsing, and multi-model aggregator.
* `corrupted_gol_benchmark.csv` - The official dataset of 600 mutated tasks across 3 difficulty tiers (Easy, Medium, Hard).

## 📊 Key Findings

Evaluations of frontier models reveal a near-universal collapse in reasoning when forced to override strong training priors. While models score >95% on standard Game of Life tasks, they fail catastrophically when the rules change.

**Overall Rule Override Score (ROS):**
* **Google Gemini 2.5 Flash:** 31.00%
* **Anthropic Claude 3.5 Sonnet:** 6.33%
* **Alibaba Qwen 3 (235B):** 5.17%

**The Pattern Trap:** Models performed worst on purely random grids (0.0% on Medium/Hard). However, on grids featuring named classic patterns (e.g., "Glider"), models scored between 20% and 54%. Analysis reveals models anchored to their memorized trajectories. For mild mutations, this accidentally inflated accuracy. For aggressive mutations, it caused total failure. **This proves spatial logic accuracy in current models is heavily driven by prior strength, not fluid reasoning.**

## 🚀 How to Use

### Running the Benchmark
The evaluation script is designed to run within the `kaggle_benchmarks` SDK ecosystem. 

1. Clone the repository.
2. Load the `.csv` dataset into your Kaggle Notebook environment.
3. Execute the `AlgoOverride_Benchmark.ipynb` script.
4. The script uses a robust two-stage regex pipeline to extract strictly formatted coordinate lists (e.g., `[(1, 2), (3, 4)]`) from chatty LLM outputs, ensuring objective, LLM-as-a-judge-free grading.

### Generating Custom Datasets
You can use `dataset_generator.py` to create your own corrupted cellular automata grids. The generator allows you to configure grid sizes, step depth, and specific rule mutations (e.g., Von Neumann neighborhoods, toroidal wraps, altered survival thresholds).

## 📝 Citation
If you use this benchmark or dataset in your research, please link back to this repository.

*This project was developed as a submission for the Kaggle "Measuring Progress Toward AGI" Hackathon (April 2026).*
