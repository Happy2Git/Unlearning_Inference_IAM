# Towards Lifecycle Unlearning Commitment Management: Measuring Sample-level Unlearning Completeness

**This repository contains the official implementation and resources for the paper "Towards Lifecycle Unlearning Commitment Management: Measuring Sample-level Unlearning Completeness," accepted at Usenix Security 2025.**

---

## 🎯 Project Overview

This project introduces **Interpolated Approximate Measurement (IAM)**, a novel framework for efficiently and effectively measuring the completeness of machine unlearning at the sample level. As data privacy regulations become more stringent, the ability to reliably remove specific data points from trained models is crucial. IAM addresses the limitations of existing unlearning evaluation methods, offering a scalable and nuanced approach to assessing unlearning.

---

## ⚠️ The Challenge with Current Unlearning Evaluation

Traditional methods for verifying machine unlearning, such as Membership Inference Attacks (MIAs), face significant hurdles:

* **Computational Cost:** Achieving high MIA effectiveness often demands resources that can exceed the cost of retraining the model from scratch.
* **Granularity:** MIAs are primarily binary (sample in or out) and struggle to quantify the degree of unlearning, especially in approximate unlearning scenarios.

---

## ✨ Introducing IAM: Interpolated Approximate Measurement

IAM offers a more efficient and granular solution by:

* **Natively designing for unlearning inference:** It's built from the ground up to evaluating unlearning.
* **Quantifying sample-level unlearning completeness:** It measures how thoroughly a specific sample has been unlearned.
* **Interpolating generalization-fitting gap:** It analyzes the model's behavior on queried samples to infer unlearning.
* **Scalability:** It can be applied to Large Language Models (LLMs) using just **one pre-trained shadow model**.

---

## 🔑 Key Features of IAM

* **Strong Binary Inclusion Test Performance:** Effective for verifying exact unlearning.
* **High Correlation for Approximate Unlearning:** Accurately reflects the degree of unlearning in approximate methods.
* **Efficiency:** Significantly reduces computational overhead compared to traditional MIA approaches.
* **Theoretical Backing:** Supported by theoretical analysis of its scoring mechanism.

---

## 📈 Practical Applications and Findings

By applying IAM to recent approximate unlearning algorithms, our research has uncovered:

* General risks of **over-unlearning**: Where more than the intended information is removed, potentially harming model utility.
* General risks of **under-unlearning**: Where the targeted data is not sufficiently removed, posing privacy risks.

These findings highlight the critical need for robust safeguards in approximate unlearning systems, a role IAM can help fulfill.

---

## 🚀 Getting Started

### Prerequisites

* **Conda/Mamba:** This environment is set up using [Mamba](https://github.com/conda-forge/miniforge#mambaforge) (or Conda). Ensure you have Mambaforge or Miniconda/Anaconda installed.
* **CUDA-enabled GPU:** Required for PyTorch with GPU support (`cu124`). Ensure your NVIDIA drivers are compatible.
* **Linux-based OS:** The provided setup instructions are for a `linux-64` platform.

### Environment Setup

The following steps will help you create the `UnInf_IAM` conda environment and install the necessary dependencies.

1.  **Create and activate the conda environment:**
    We recommend using Mamba for faster environment creation.

    ```bash
    # Using Mamba (recommended)
    mamba create -n UnInf_IAM python=3.12.5 requests -y
    conda activate UnInf_IAM
    ```
    If you don't have Mamba, you can use Conda:
    ```bash
    # Using Conda
    conda create --name UnInf_IAM python=3.12.5 requests -y
    conda activate UnInf_IAM
    ```

2.  **Install PyTorch with CUDA 12.4 support:**

    ```bash
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
    ```

3.  **Install other Python packages:**

    ```bash
    pip install pandas==2.2.2 torch_optimizer==0.3.0 scikit-learn==1.5.1 scipy==1.14.1 xgboost==2.1.1
    pip install matplotlib seaborn==0.13.2 wget==3.2 ipykernel==6.29.5 ipywidgets==8.1.3 jupyterlab_widgets==3.0.11 tqdm==4.66.5
    ```

4.  **Set the `LD_LIBRARY_PATH`:**
    This step is crucial for PyTorch to correctly find NVIDIA libraries. **You'll need to adjust the path `~/mambaforge/` if your Mambaforge/Miniconda installation is elsewhere.** This command should be run every time you activate the environment, or you can add it to your shell's startup script (e.g., `.bashrc`, `.zshrc`).

    ```bash
    export LD_LIBRARY_PATH=~/mambaforge/envs/UnInf_IAM/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
    ```
    To find the correct path for `nvjitlink` within your specific `UnInf_IAM` environment if you didn't install mambaforge in `~/mambaforge/`, you can use:
    ```bash
    CONDA_PREFIX_PATH=$(conda info --base)
    # Or if mamba is not in the base env, find the UnInf_IAM prefix directly
    # CONDA_PREFIX_PATH=$(conda env list | grep UnInf_IAM | awk '{print $NF}')
    # Be careful with the above if you have multiple envs with similar names.
    # A safer way is to activate the env and then use:
    # conda activate UnInf_IAM
    # PYTHON_SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')
    # export LD_LIBRARY_PATH=${PYTHON_SITE_PACKAGES}/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

    # A more robust way after activating the environment:
    # conda activate UnInf_IAM
    # NVJITLINK_LIB_PATH=$(python -c "import os; import nvidia.nvjitlink as nv; print(os.path.dirname(nv.__file__))")/lib
    # export LD_LIBRARY_PATH=${NVJITLINK_LIB_PATH}:${LD_LIBRARY_PATH}
    # printenv LD_LIBRARY_PATH # To verify
    ```
    For simplicity, assuming `mambaforge` is in the home directory as per the original script:
    ```bash
    # Make sure to run this in the terminal where you will be running your code
    # or add it to your ~/.bashrc or ~/.zshrc for persistence.
    # Replace '~/mambaforge/' with the actual path to your mambaforge/miniconda installation if different.
    export LD_LIBRARY_PATH=$HOME/mambaforge/envs/UnInf_IAM/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
    ```

### Usage

#### Pre-training Shadow OUT Model(s)
Typically, we recommend pre-training 10 shadow out models per dataset to obtain a reliable average across different trials, with each trial using a different shadow out model. However, for a quick setup or initial testing, training just one shadow out model is also acceptable, though this does not account for variance across shadow models.
You can start training with the following command:
```python
python shadow_training.py --dataname 'cifar100' --ratio 0.8 --shadow_nums 1 --origin True --VERBOSE True
```

#### Original Model and Unlearned Model
We generate 10 distinct retain/unlearn splits by varying the random seed from 42 to 51. For each split, a model is retrained on the corresponding retained subset to perform exact unlearning.

For random sample unlearning:
```python
python new_reseed.py  --SEED_init=42 --LOOP=1 --unlearn_type=set_random --dataname=cifar100 --model_numbs=1
```
For partial class unlearning:
```python
python new_reseed.py  --SEED_init=42 --LOOP=1 --unlearn_type=class_percentage --dataname=cifar100 --model_numbs=1
```

#### Binary Unlearning Inference (BinUI)
Then evaluate using:
```python
python new_test_scores.py --LOOP=1 --unlearn_type='set_random' --model_numbs=1 --dataname='cifar100'
```
> **Note:** If you have 10 pre-trained shadow models, set `--LOOP=10  --SHADOW_AVE_FLAG` to compute the average result over 10 trials.

### 📄 Reproducing Results from the Paper

We pretrained **128 shadow models** for more reliable averages.

#### Files and Corresponding Experiments

| Script                          | Description                                                 | Reference         |
| -------------------------------|-------------------------------------------------------------|-------------------|
| `new_records.py`                | Binary Unlearning Inference (BinUI)                        | Table 1, 10; Figure 1 |
| `new_records.slurm`            | SLURM job version of `new_records.py` for BinUI of (partial) class unlearning | Table 1, 10; Figure 1 |
| `new_records_refnums.slurm`    | SLURM job version of `new_records.py` for BinUI with random sample unlearning and varying shadow numbers | Table 1; Figures 1, 5 |
| `new_records_refnums_slurm_gen.py`    | SLURM job version of `new_records.py` for BinUI with random sample unlearning and varying shadow numbers | Table 1; Figures 1, 5 |
| `new_records_internal.py`      | Score-based Unlearning Inference (ScoreUI)                  | Table 2           |
| `new_records_internal_slurm_gen.py`      | Score-based Unlearning Inference (ScoreUI)        | Table 2           |
| `new_records_shift.py`         | BinUI for CIFAR-10 and CINIC-10 (OOD scenarios)             | Table 3           |
| `new_records_shift.slurm`      | BinUI for CIFAR-10 and CINIC-10 (OOD scenarios)             | Table 3           |
| `new_records_internal_shift.py`| ScoreUI for CIFAR-10 and CINIC-10 (OOD scenarios)           | Table 3           |
| `new_records_internal_shift.slurm`| ScoreUI for CIFAR-10 and CINIC-10 (OOD scenarios)        | Table 3           |
| `new_records_shift_model.py`   | BinUI on CIFAR-100 using different architectures            | Figure 4          |
| `new_test_incremental.py`      | BinUI on dynamic training datasets                          | Table 4           |
| `new_test_scores.py`           | BinUI on CIFAR-100 using different scoring functions        | Table 5           |
| `new_records_methods_gen.py`   | Benchmarking approximate unlearning methods                 | Table 8           |
| `new_records_methods_class_gen.py`   | Benchmarking approximate unlearning methods           | Table 8           |
| `mia_llms_benchmark`           | LLM unlearning^1 | Tables 6, 7        |
| `new_records_plot.ipynb`       | Generating LaTeX-formatted tables and figures^2             | Tables 1–14; Figures 1, 4, 5 |

^1 We implement LiRA-On, RMIA-On, and IAM-On for LLMs based on the [mia_llms_benchmark](https://github.com/computationalprivacy/mia_llms_benchmark)  framework. To run the related scripts, please follow the instructions in the README.md of [mia_llms_benchmark](https://github.com/computationalprivacy/mia_llms_benchmark) and activate the 'mia_llms_benchmark' conda environment.

^2 We’ve uploaded the necessary logs for generating the tables and figures in `new_records_plot.ipynb`. You're also free to retrain and reproduce all results using the scripts above, or just run `run_all.sh` if everything's set up.

---

## 📜 Citation

If you use IAM or this codebase in your research, please cite our Usenix Security 2025 paper:

```bibtex
@inproceedings{yourlastname2025iam,
  title={Towards Lifecycle Unlearning Commitment Management: Measuring Sample-level Unlearning Completeness},
  author={Cheng-Long Wang, Qi Li, Zihang Xiang, Yinzhi Cao, Di Wang},
  booktitle={Proceedings of the 34th USENIX Security Symposium (USENIX Security 25)},
  year={2025},
  publisher={USENIX Association}
}
```

---

## 🤝 Contributing

We welcome contributions to improve IAM! Please feel free to submit issues or pull requests.

---

## 📄 License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
---
