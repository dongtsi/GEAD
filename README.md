<div align=center><img src="doc/gead_logo.png" width="80%"></div>

> *Rules Refine the Riddle: Global Explanation for Deep Learning-Based Anomaly Detection in Security Applications.* Accepted by [CCS'24](https://www.sigsac.org/ccs/CCS2024).
**Please download the relased code from [here](https://github.com/dongtsi/GEAD/releases/tag/AEC) if you want to reproduce the results in our paper.**

![](https://img.shields.io/badge/license-MIT-green.svg)
![](https://img.shields.io/badge/language-python-blue.svg)
![](https://img.shields.io/badge/framework-pytorch-red.svg)

### [<span style="color:blue">【中文README（开源网安计划）请点击此处】</span>](./README_CN.md)


## Introduction
This artifact is the implementation and experiment results of GEAD proposed in CCS'24.
In short, GEAD is a method for extracting rules in deep learning-based anomaly detection models. As shown in the figure below, it contains several core steps:
![GEAD Overview](./doc/gead_overview.png)
- **Root regression tree generation**: leveraging black-box knowledge distillation methods to extract a raw rule tree
- **Low-confidence Region Identification**: Find regions that cause inconsistencies between the original model and the tree-based explainable model
- **Low-confidence augmentation**: augmenting data that can lead to inconsistent decisions between the two models
- **Low-confidence rule generation**: using augmented data to expand original rules
- **Tree merging and discretization**: simplifying the rules to increase the readability for operators
- **Rule generation (optional)**: Convert the rule tree into readable a rule set


## Code Structure
The following is a brief introduction to the directory structure of this artifact:
```
- baseline/         ; code of baselines
- code/
    - gead.py       ; code of GEAD
    - gead_seq/py   ; code of GEAD (for RNN)
    - ...
- demo/ 
    - demo.ipynb    ; demo to show how to use GEAD
    - ...
- experiment/
    - results/  ;reproduced experient results
    - Fidelity_Evaluation.ipynb ; experiment 1
    - Usage 2.ipynb             ; experiment 2
- setup/        ;environment setup files   
- doc/              ; images used in README
- README.md         ; instructions of this artifact
```


## Environment Setup
> This implementation has been successfully tested in **Ubuntu 16.04** server with **Python 3.7.16**.
To ensure compatibility, this artifact (pytorch-based parts) can be fully run with **CPU** (GPU/CUDA is not required).

To ensure the proper functioning of this artifact, please follow the commands below:

1. Ensure that you have `conda` installed on your system. If you do not have `conda`, you can install it as part of the Anaconda distribution or Miniconda.
2. Open a terminal or command prompt.
3. Create a new conda environment with the name of your choice (e.g., `GEAD`) and specify the version of python to configure it:
   ```bash
   conda create -n GEAD python=3.7.16
   ```
4. Once the environment is created, activate it by running:
   ```bash
   conda activate GEAD
   ```
   This will switch your command line environment to use the newly created conda environment with all the necessary packages.
5. Run the following command to install all the required packages:
   ```bash
   pip install -r setup/requirements.txt
   ```
   This command tells `pip` to install all the packages listed in the `requirements.txt` file.

Below, the experiments or demos in this artifact mainly use [jupyter notebook](https://jupyter.org/). So make sure you can view and execute the notebook (.ipynb) files. 
How to use jupyter notebook can be found on the [official website](https://docs.jupyter.org/en/latest/). 
In short, **select the right kernel (namely, the above `GEAD`) and then execute all cells (except markdown cells) in sequence**. All cells in this artifact have been pre-executed with output shown. If all goes well, you should get consistent output in your environment.



## Demo
We provide **[a step-by-step demo](demo/demo.ipynb)** of explaining an autoencoder-based anomaly detection model with GEAD, which is also the result in Section 4.3.1 of our paper.