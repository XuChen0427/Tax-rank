# Tax-rank

Tax-rank implementation for SIGIR2024.



## Model

To run the Tax-rank model, you need to run the `run_tax-rank.py`.(with default setting)

```
python run_tax-rank.py
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

Result should be:
```
GINI:0.968  ACC:6.266
```

For entire dataset, please download from urls from paper.
