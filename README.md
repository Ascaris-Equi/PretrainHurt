# Pre-Trained Models May Hurt Immunogenicity Prediction Performance

This repository contains the code and dataset used in the study titled "Pre-Trained Models May Hurt Immunogenicity Prediction Performance" by Jiahao Ma, Hongzong Li, Jian-Dong Huang, Ye-Fan Hu, and Yifan Chen. The study investigates the impact of using pre-trained models in the task of Peptide-MHC-TCR interaction prediction, and demonstrates that pre-trained TCR embeddings may lead to suboptimal performance in certain scenarios.

## Abstract

In this work, we explore whether pre-trained models can hurt the predictive performance of immunogenicity tasks, specifically Peptide-MHC-TCR interaction predictions. We evaluate a modified pMTnet model (`pMTnet-NoPreCA`), which does not use pre-trained TCR embeddings, and compare its performance against the original pMTnet. Our results reveal that the `pMTnet-NoPreCA` model outperforms the original when pre-trained embeddings are excluded, suggesting that pre-trained models may not always be effective for this specific task.

## Installation

To use the code in this repository, first clone the repository:

```bash
git clone https://github.com/Ascaris-Equi/PretrainHurt.git
cd PretrainHurt
```
Then, run with the following codes:
```bash
jupyter nbconvert --to notebook --execute pmtnet-torch.ipynb
jupyter nbconvert --to notebook --execute nopreca-torch.ipynb
```
## Results

Our experiments show that the pMTnet-NoPreCA model significantly outperforms the original pMTnet model in both ROC and Precision-Recall metrics:

|  Metric           |pMTnet (pMTnet Dataset)| pMTnet-NoPreCA (pMTnet Dataset)|pMTnet (OOD Dataset)| pMTnet-NoPreCA (OOD Dataset)|
|-------------------|-------------:|-------------:|-------------:|-------------:|
| ROC AUC           |       0.7327 |       0.9168 |       0.4722 |       0.7126 |
| PR AUC            |       0.7065 |       0.8912 |       0.4832 |       0.7031 |

One dataset used for this study is available in the original paper{https://github.com/tianshilu/pMTnet}, designed as **pMTnet Dataset**.  
Another dataset used here is obtained from VDJdb{https://vdjdb.cdr3.net/}, designed as **OOD Dataset**. 
## Acknowledgments

This research was supported by the National Key Research and Development Program of China (2021YFA0910700), the Health and Medical Research Fund, and other funding bodies. For a full list of acknowledgments, please refer to the paper.

## License

This project is licensed under the MIT License.

## Notes
- Make sure you have a requirements.txt file listing all the dependencies (e.g., PyTorch, NumPy, etc.).
- Ensure that the paths in the README.md match the actual structure of your repository.
- You can include any additional information, such as specific hardware requirements or further instructions for running experiments, as needed.
