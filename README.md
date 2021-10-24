#        Code Repository for the Paper
## "Identification of the Generalized Condorcet Winner in Multi-dueling Bandits"
       (To appear in: Proceedings of NeurIPS2021)

The code is written in Python 3.7.

You can cite our paper as follows:

```
@inproceedings{Haddenhorst2021,
  title={Identification of the Generalized Condorcet Winner in Multi-dueling Bandits},
  author={Haddenhorst, Bj{\"o}rn and Bengs, Viktor and H{\"u}llermeier, Eyke},
  booktitle = {Proceedings of Advances in Neural Information Processing Systems 34 (NeurIPS 2021)},
  year={2021},
}
```
## Requirements
To install requirements:

```setup
pip install -r requirements.txt
```

## Evaluation
- (A) To obtain the evaluation results of the algorithms, uncomment the corresponding code in "Neurips2021_experiments.py" and execute it.
- (B) To repeat the empirical comparison of the two lower bounds (Prop. 4.1 and Thm 5.2) for the single 
bandit case (m=k), simply execute "NeurIPS_LB_comparison.py".

## Results
- After repeating all experiments in (A), the results shown in the tables are written saved the following files

|  TABLE(S) | FILE  |
|---|---|
| 2  | Experiment_PW_m5.txt  | 
| 3,6,7  | Experiment1_m5_gamma_005.txt  | 
| 3,6,7  | Experiment1_m10_gamma_005.txt   |  
|  4 | Experiment_PW_m10.txt  | 
| 5  |  Experiment_PW_PWData.txt         |
| 6,7  | Experiment1_m15_gamma_005.txt  | 
| 6,7  | Experiment1_m20_gamma_005.txt   |
| 8  | Experiment_DKWT_vs_algo5_v1.txt  | 
| 9   | Experiment_DKWT_vs_algo5_v2.txt  | 

- The results for (B) are only shown in the terminal and not saved to any file.

In case of any questions, please contact Björn Haddenhorst (bjoernha@mail.upb.de).
