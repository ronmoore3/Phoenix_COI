from .globalvars import plot_colors
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
import shap


def prc_plot(results, save_path):
    '''
    Create Precision-Recall plots
    '''
    os.makedirs(save_path, exist_ok=True)

    color_vals = list(plot_colors.values())
    recall_base = np.linspace(0, 1, 101)

    fig, ax = plt.subplots(figsize=(30,30))
    model_keys = sorted(results.keys())

    # calculate point for fixed recall and fixed precision
    dist = recall_base - 0.85
    dist = np.where(dist < 0, np.inf, dist)
    optimal_idx = dist.argmin()
    fixed_recall = recall_base[optimal_idx]
    fixed_precision = np.inf
    auprc_max = 0
    for i, model in enumerate(model_keys):
        if model == 'both':
            model_name = 'EMR + COI'
        else:
            model_name = model.upper()
        model_results = results[model]
        precisions = []
        trues = []
        probs = []
        # get number of trials
        prob_cols = [x for x in model_results.columns if 'y_prob' in x]
        true_cols = [x for x in model_results.columns if 'y_true' in x]
        trials = len(prob_cols)
        auprcs = []
        for trial in range(trials):
            if len(true_cols) > 1:
                y_prob = model_results.loc[~model_results[f'y_prob_{trial}'].isna(), f'y_prob_{trial}'].values
                y_true = model_results.loc[~model_results[f'y_true_{trial}'].isna(), f'y_true_{trial}'].values
            else:
                y_prob = model_results[f'y_prob_{trial}'].values
                y_true = model_results['y_true'].values
            auprc = average_precision_score(y_true, y_prob)
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            precision_interp = np.interp(recall_base, precision, recall)
            precision_interp[0] = 1
            precisions.append(precision_interp)
            trues.append(y_true)
            probs.append(y_prob)
            auprcs.append(auprc)
        
        # calculate mean and standard deviation
        true_arr = np.concatenate(trues)
        precisions = np.array(precisions)
        precision_mean = np.mean(precisions, axis=0)
        precision_std = np.std(precisions, axis=0)
        precision_upper = precision_mean + precision_std
        precision_lower = precision_mean - precision_std

        # calculate 95% CI for auprc
        auprc_mean = np.mean(auprcs)
        ci = st.norm.interval(confidence=0.95, loc=auprc_mean, scale=st.sem(auprcs))
        auprc_upper = auprc_mean if np.isnan(ci[1]) else ci[1]
        auprc_lower = auprc_mean if np.isnan(ci[0]) else ci[0]

        # find recall (sensitivity) relating to largest auprc
        if auprc_mean > auprc_max:
            auprc_max = auprc_mean
            fixed_precision = precision_mean[optimal_idx]

        if i == 0:
            no_skill_mean = len(true_arr[true_arr==1]) / len(true_arr)
            ax.plot([0, 1], [no_skill_mean, no_skill_mean], linestyle='--', color='black', label='No Skill')
        ax.plot(recall_base, precision_mean, lw=3, color=color_vals[i], label=f"{model_name} AUPRC: {auprc_mean:.2f} ({auprc_lower:.2f}, {auprc_upper:.2f})")
        ax.fill_between(recall_base, precision_lower, precision_upper, color=color_vals[i], alpha=0.2)

    ax.scatter(fixed_recall, fixed_precision, marker='+', s=200**2, color='black', linewidths=3)
    ax.tick_params(labelsize=36)
    ax.set_xlabel('Recall', fontsize=48)
    ax.set_ylabel('Precision', fontsize=48)
    ax.legend(loc='upper right', fontsize=36)
    plt.tight_layout()
    fig.savefig(f"{save_path}/prc.png", dpi=300)
    plt.close()


def roc_plot(results, save_path):
    '''
    Create ROC plots
    '''
    os.makedirs(save_path, exist_ok=True)

    color_vals = list(plot_colors.values())
    fpr_base = np.linspace(0, 1, 101)

    # TODO: plot mean and standard deviation for each model
    fig, ax = plt.subplots(figsize=(30,30))
    ax.plot([0,1], [0,1], linestyle='--', color='black', label='No Skill')
    model_keys = sorted(results.keys())

    fixed_fpr = np.inf
    fixed_tpr = np.inf
    auroc_max = 0
    for i, model in enumerate(model_keys):
        if model == 'both':
            model_name = 'EMR + COI'
        else:
            model_name = model.upper()

        model_results = results[model]
        tprs = []
        # get number of trials
        prob_cols = [x for x in model_results.columns if 'y_prob' in x]
        true_cols = [x for x in model_results.columns if 'y_true' in x]
        trials = len(prob_cols)
        aurocs = []
        for trial in range(trials):
            if len(true_cols) > 1:
                y_prob = model_results.loc[~model_results[f'y_prob_{trial}'].isna(), f'y_prob_{trial}'].values
                y_true = model_results.loc[~model_results[f'y_true_{trial}'].isna(), f'y_true_{trial}'].values
            else:
                y_prob = model_results[f'y_prob_{trial}'].values
                y_true = model_results['y_true'].values
            auroc = roc_auc_score(y_true, y_prob)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            tpr_interp = np.interp(fpr_base, fpr, tpr)
            tpr_interp[0] = 0
            tprs.append(tpr_interp)
            aurocs.append(auroc)

        # calculate mean and standard deviation
        tprs = np.array(tprs)
        tpr_mean = np.mean(tprs, axis=0)
        tpr_std = np.std(tprs, axis=0)
        tpr_upper = tpr_mean + tpr_std
        tpr_lower = tpr_mean - tpr_std

        # calculate 95% CI for auroc
        auroc_mean = np.mean(aurocs)
        ci = st.norm.interval(confidence=0.95, loc=auroc_mean, scale=st.sem(aurocs))
        auroc_upper = auroc_mean if np.isnan(ci[1]) else ci[1]
        auroc_lower = auroc_mean if np.isnan(ci[0]) else ci[0]

        # find tpr (sensitivity) relating to largest auroc
        if auroc_mean > auroc_max:
            auroc_max = auroc_mean

            # find tpr (sensitivity) closest to 0.85
            dist = tpr_mean - 0.85
            dist = np.where(dist < 0, np.inf, dist)
            optimal_idx = dist.argmin()

            fixed_fpr = fpr_base[optimal_idx]
            fixed_tpr = tpr_mean[optimal_idx]

        ax.plot(fpr_base, tpr_mean, lw=3, color=color_vals[i], label=f"{model_name} AUROC: {auroc_mean:.2f} ({auroc_lower:.2f}, {auroc_upper:.2f})")
        ax.fill_between(fpr_base, tpr_lower, tpr_upper, color=color_vals[i], alpha=0.2)

    ax.scatter(fixed_fpr, fixed_tpr, marker='+', s=200**2, color='black', linewidths=3)
    ax.tick_params(labelsize=36)
    ax.set_xlabel('False Positive Rate', fontsize=48)
    ax.set_ylabel('True Positive Rate', fontsize=48)
    ax.legend(loc='lower right', fontsize=36)
    plt.tight_layout()
    fig.savefig(f"{save_path}/roc.png", dpi=300)
    plt.close()


def calibration_plot(results, save_path, n_bins=10):
    '''
    Create calibration plots
    '''
    os.makedirs(save_path, exist_ok=True)

    color_vals = list(plot_colors.values())
    prob_pred_base = np.linspace(0, 1, 101)

    fig, ax = plt.subplots(figsize=(30,30))
    ax.plot([0,1], [0,1], linestyle='--', color='black', label='Perfect Calibration')
    model_keys = sorted(results.keys())

    for i, model in enumerate(model_keys):
        if model == 'both':
            model_name = 'EMR + COI'
        else:
            model_name = model.upper()

        model_results = results[model]
        prob_trues = []
        # get number of trials
        prob_cols = [x for x in model_results.columns if 'y_prob' in x]
        true_cols = [x for x in model_results.columns if 'y_true' in x]
        trials = len(prob_cols)
        for trial in range(trials):
            if len(true_cols) > 1:
                y_prob = model_results.loc[~model_results[f'y_prob_{trial}'].isna(), f'y_prob_{trial}'].values
                y_true = model_results.loc[~model_results[f'y_true_{trial}'].isna(), f'y_true_{trial}'].values
            else:
                y_prob = model_results[f'y_prob_{trial}'].values
                y_true = model_results['y_true'].values
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
            prob_true_interp = np.interp(prob_pred_base, prob_pred, prob_true)
            prob_true_interp[0] = 0
            prob_trues.append(prob_true_interp)
        
        # calculate mean and standard deviation
        prob_trues = np.array(prob_trues)
        prob_true_mean = np.mean(prob_trues, axis=0)     
        prob_true_std = np.std(prob_trues, axis=0)     
        prob_true_upper = prob_true_mean + prob_true_std
        prob_true_lower = prob_true_mean - prob_true_std
        ax.plot(prob_pred_base, prob_true_mean, lw=3, color=color_vals[i], label=model_name)
        ax.fill_between(prob_pred_base, prob_true_lower, prob_true_upper, color=color_vals[i], alpha=0.2)
        
    ax.tick_params(labelsize=36)
    ax.set_xlabel('Mean Predicted Probability', fontsize=48)
    ax.set_ylabel('Fraction of Positives', fontsize=48)
    ax.legend(loc='upper left', fontsize=36)
    plt.tight_layout()
    fig.savefig(f"{save_path}/calibration.png", dpi=300)
    plt.close()


def shap_plot(shap_vals, save_path):
    '''
    Create SHAP beeswarm plots
    '''
    os.makedirs(save_path, exist_ok=True)
    for model in sorted(shap_vals.keys()):
        model_shap_vals = shap_vals[model][:,:,1] if 'RF' in model else shap_vals[model]
        shap.plots.beeswarm(model_shap_vals,
                            max_display=20,
                            show=False
                            )
        # Modifying main plot parameters
        fig, ax = plt.gcf(), plt.gca()
        ax.tick_params(labelsize=24)
        ax.set_xlabel("SHAP value (impact on model output)", fontsize=30)
        # Get colorbar and modify parameters
        cb_ax = fig.axes[1] 
        cb_ax.tick_params(labelsize=24)
        cb_ax.set_ylabel('Feature value', fontsize=30)
        fig.set_size_inches(15,10)
        plt.tight_layout()
        plt.savefig(f"{save_path}/{model}.png", dpi=300)
        plt.clf()
    plt.close()
