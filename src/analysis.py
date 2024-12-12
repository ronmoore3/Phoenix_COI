import numpy as np
import os
import pandas as pd
import pickle
import shap
import scipy.stats as st
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from src.utils import get_predictions, expected_calibration_error


def get_metrics(results):
    '''
    Write model's performance metric values to a csv file
    '''
    metrics = {'specificity': [],
               'sensitivity': [],
               'precision': [],
               'fpr': [],
               'accuracy': [],
               'f1': [],
               'auroc': [],
               'auprc': []
               }
    prob_cols = [x for x in results.columns if 'y_prob' in x]
    true_cols = [x for x in results.columns if 'y_true' in x]
    trials = len(prob_cols)
    for trial in range(trials):
        if len(true_cols) > 1:
            y_prob = results.loc[~results[f'y_prob_{trial}'].isna(), f'y_prob_{trial}'].values
            y_true = results.loc[~results[f'y_true_{trial}'].isna(), f'y_true_{trial}'].values
        else:
            y_prob = results[f'y_prob_{trial}'].values
            y_true = results['y_true'].values

        y_pred = get_predictions(y_prob, y_true, sensitivity=0.85)
        # y_pred = get_predictions(y_prob)
        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1]).ravel()
        
        spec = tn/(tn+fp) 
        sens = tp/(tp+fn)
        prec = tp/(tp+fp)
        fpr = fp/(fp+tn)
        acc = (tp+tn)/(tp+fp+fn+tn)

        f1 = f1_score(y_true, y_pred, labels=[0,1])    
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)

        # probably will ignore ECE for now
        # expected calibration error
        # ece = expected_calibration_error(y_prob, y_true, bins=5)
        
        metrics['specificity'].append(spec)
        metrics['sensitivity'].append(sens)
        metrics['precision'].append(prec)
        metrics['fpr'].append(fpr)
        metrics['accuracy'].append(acc)
        metrics['f1'].append(f1)
        metrics['auroc'].append(auroc)
        metrics['auprc'].append(auprc)

    return metrics


def save_metrics(metrics, save_path, model_name):
    '''
    Write model's performance metric values to a csv file
    '''
    os.makedirs(save_path, exist_ok=True)
    for key in metrics.keys():
        mean = np.mean(metrics[key])
        ci = st.norm.interval(confidence=0.95, loc=mean, scale=st.sem(metrics[key]))
        ci_upper = mean if np.isnan(ci[1]) else ci[1]
        ci_lower = mean if np.isnan(ci[0]) else ci[0]
        metric_val = f"{np.round(mean, 2):.2f} ({np.round(ci_lower, 2):.2f}, {np.round(ci_upper, 2):.2f})"
        metrics[key] = metric_val

    if os.path.isfile(f"{save_path}/metrics.csv"):
        metrics_df = pd.read_csv(f"{save_path}/metrics.csv")
        if model_name in metrics_df['model'].unique():
            metrics_df.loc[metrics_df['model'] == model_name, list(metrics.keys())] = list(metrics.values())
        else:
            metrics_df.loc[len(metrics_df)] = [model_name] + list(metrics.values())
    else:
        metrics_df = pd.DataFrame.from_dict({k:[v] for k,v in metrics.items()})
        metrics_df.insert(loc=0, column='model', value=model_name)
    
    metrics_df.sort_values(by='model', inplace=True)

    metrics_df.to_csv(f"{save_path}/metrics.csv", index=False)


def get_shap_values(model, train_data, test_data, test_data_unscaled, save_path, model_name):
    '''
    Save model's predictions to csv file for plotting
    '''
    os.makedirs(save_path, exist_ok=True)
    explainer = shap.LinearExplainer(model, train_data) if model_name == 'LR' else shap.TreeExplainer(model)
    shap_vals = explainer(test_data)
    shap_vals.data = test_data_unscaled

    with open(f'{save_path}/{model_name}.pkl', 'wb') as f:
        pickle.dump(shap_vals, f)


def save_model_results(results, save_path, model_name):
    '''
    Save model's predictions to csv file for plotting
    '''
    os.makedirs(save_path, exist_ok=True)
    results = pd.DataFrame(results)
    results.to_csv(f"{save_path}/{model_name}.csv", index=False)