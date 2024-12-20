import os
import pandas as pd
import pickle
from src.plot import calibration_plot, prc_plot, roc_plot, shap_plot

"""
Plot results for models
"""


def model_plots(results, df, save_path):
    cohort_results = results
    cohort_csns = df['csn'].unique()
    for model in results.keys():
        model_results = results[model]
        prob_cols = [x for x in model_results.columns if 'y_prob' in x]
        trials = len(prob_cols)
        cohort_trial_results = []
        for trial in range(trials):
            cohort_trial_cond = model_results[f'csn_{trial}'].isin(cohort_csns)
            cohort_trial_info = [f'csn_{trial}', f'y_true_{trial}', f'y_prob_{trial}']
            cohort_trial_results.append(model_results.loc[cohort_trial_cond, cohort_trial_info].reset_index(drop=True))
        cohort_trial_results_df = pd.concat(cohort_trial_results, axis=1)
        cohort_trial_results_df.columns = model_results.columns
        cohort_results[model] = cohort_trial_results_df

    # plot results
    for plot in ['calibration', 'prc', 'roc']:
        if plot == 'calibration':
            calibration_plot(cohort_results, save_path)
        elif plot == 'prc':
            prc_plot(cohort_results, save_path)
        elif plot == 'roc':
            roc_plot(cohort_results, save_path)


def shap_plots(shap_vals, results, df, save_path):
    # for cohort in ['suspected_infection', 'sepsis', 'septic_shock']:
    for cohort in ['suspected_infection']:
        if cohort == 'suspected_infection':
            cohort_shap_vals = shap_vals
        else:
            cohort_shap_vals = {}
            cohort_csns = df.loc[df[cohort] == 1, 'csn'].unique()
            for model in results.keys():
                model_name = f'{cohort}_{model}'
                model_results = results[model]
                model_shap_vals = shap_vals[model]
                true_cols = [x for x in model_results.columns if 'y_true' in x]
                trials = len(true_cols)
                # internal or external validation
                csn_col = f'csn_{trials-1}' if trials > 1 else 'csn'
                cohort_cond = model_results[csn_col].isin(cohort_csns)
                cohort_ind = model_results.loc[cohort_cond].index.values
                cohort_shap_vals[model_name] = model_shap_vals[cohort_ind]
        # plot results
        shap_plot(cohort_shap_vals, save_path)


if __name__ == '__main__':
    data_path = 'data/dataset_24h_filtered.parquet.gzip'
    df = pd.read_parquet(data_path)
    for train_data in ['EG', 'SR']:
        results_path = f"{os.getcwd()}/results/{train_data}"
        print(f"Creating plots for {train_data}...")
        results = {}
        for validation in ['internal', 'external']:
            validation_path = f'{results_path}/model_results/{validation}'
            plots_path = f'{results_path}/plots'
            model_plots_path = f'{plots_path}/models/{validation}'
            for csv_file in sorted(os.scandir(validation_path), key=lambda x: x.name):
                model_name = f"{csv_file.name.split('.csv')[0]}"
                results[model_name] = pd.read_csv(csv_file.path)

            shap_vals = {}
            shap_data_path = f'{results_path}/shap_data'
            shap_plots_path = f'{plots_path}/shap/{validation}'
            for pkl_file in os.scandir(f'{shap_data_path}/{validation}'):
                if pkl_file.is_file():
                    model_name = pkl_file.name.split('.pkl')[0]
                    with open(pkl_file.path, 'rb') as f:
                        shap_vals[model_name] = pickle.load(f)

            # plot model and shap results
            model_plots(results, df, model_plots_path)
            shap_plots(shap_vals, results, df, shap_plots_path)
