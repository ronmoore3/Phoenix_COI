# Child Opportunity Index (COI) for Improved Health Outcomes

## About
This project looks at using social determinants of health (SDoH) called the Child Opportunity Index to improve outcomes for pediatric sepsis patients based on the novel Phoenix Sepsis criteria

## Cohort Development
For more information on how the Phoenix cohort was developed, please visit https://github.com/dchancia/ped-sepsis-prediction-ml.

## 1. Model Training
To train the model, run the script `main.py` using the following command:

```bash
python main.py --train_data --features --trials
```
### Options
**`train_data (str, default=EG)`** - Site to use for training. \
**EG** - Use Egleston for training and Scottish Rite for testing. \
**SR** - Use Scottish Rite for training and Egleston for testing.

**`features (str, default=emr)`** - Features to use for model development. \
**emr** - Use laboratory results and vital signs. \
**coi** - Use the Child Opportunity Index (COI) indicators only. \
**both** - Use the laboratory results, vital signs, and COI indicators.

**`trials (int, default=10)`** - Number of trials to use for experiment repitition.

## 2. Model Evaluation
To generate model results, run the script `sensitivity_analysis.py` using the following command: 

```bash
python sensitivity_analysis.py
```

## 3. Plot Results
To generate generate plots of the model results, run the script `plot_results.py` using the following command: 

```bash
python plot_results.py
```

## Other Info
Use the notebooks in `notebooks` to generate the other tables and figures listed in the paper.
