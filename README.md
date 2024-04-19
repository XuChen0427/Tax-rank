# A Taxation Perspective for Fair Re-ranking of SIGIR'24
## Xu Chen, Ph.D. student of Renming University of China, GSAI
Any question, please mail to xc_chen@ruc.edu.cn


## Model

To run the Tax-rank model, you need to run the `run_tax-rank.py`.(with default setting)

```
python run_tax-rank.py
```

Result should be:
```
GINI:0.968  ACC:6.266
```

If you want to change the experimental setting, you can choose to determine the parameters as followed:

| parameter | type  | range         | detail                                                       |
| --------- | ----- | :------------ | :----------------------------------------------------------- |
| U         | int   | [1 ,+inf)     | The size of users of the dataset you want to select          |
| I         | int   | [1 ,+inf)     | The size of items of the dataset you want to select          |
| mode      | str   | {'ad', 'rec'} | 'ad' if you want to run Tax-rank on the advertising dataset;  'rec' if you want to run Tax-rank on the recommendation dataset |
| topk      | int   | [1,U]         | the size of the recommendation list                          |
| t | float | [0,+inf)      | the taxation rate of Tax-rank to trade-off between accuracy and fairness |
| k         | float | (0, +inf)     | the hyperparameter to determine the size of eta              |
| lbd       | float | (0,+inf)      | the coefficient in OT projection to determine the smoothness  and the convexity of the distribution OT solution |

Here we give a example of run the model using self-determined parameter setting and a small set of dataset:

```python
python run_tax-rank.py --U=503 --I=314 --mode=rec --topk=10 --t=1 --k=0.1 --lbd=1 
```


For entire dataset, please download from urls from paper.


##For citation, please cite the following bib
```
@inproceedings{Xu-TaxRank-SIGIR24,
author = {Xu, Chen and Ye, Xiaopeng and Wang, Wenjie and Pang, Liang and Xu, Jun and Ji-rong Wen},
title = {A Taxation Perspective for Fair Re-ranking},
year = {2024},
isbn = {979-8-4007-0431-4/24/07},
publisher = {Association for Computing Machinery},
address = {Washington, DC, USA},
doi = {10.1145/3543507.3583296},
booktitle = {Proceedings of the 47th
International ACM SIGIR Conference on Research and Development in
Information Retrieval},
series = {SIGIR '24}
}
```
