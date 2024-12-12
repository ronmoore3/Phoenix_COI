"""
Get Demographic results
"""
import os
import pandas as pd
from src.analysis import get_metrics, save_metrics


def sensitivity_analysis(results, df, save_path):
    '''
    Write CSV for model's predictions by different demographics
    '''
    for model in results.keys():
        model_results = results[model]
        prob_cols = [x for x in model_results.columns if 'y_prob' in x]
        trials = len(prob_cols)
        # TODO: Focus on suspected infection for now
        for cohort in ['suspected_infection']:
        # for cohort in ['suspected_infection', 'sepsis', 'septic_shock']:
            if cohort == 'suspected_infection':
                cohort_results = model_results
            # need to filter out patients for sepsis and septic shock cohorts
            else:
                cohort_csns = df.loc[df[cohort] == 1, 'csn'].unique()
                cohort_trial_results = []
                for trial in range(trials):
                    cohort_trial_cond = model_results[f'csn_{trial}'].isin(cohort_csns)
                    cohort_trial_info = [f'csn_{trial}', f'y_true_{trial}', f'y_prob_{trial}']
                    cohort_trial_results.append(model_results.loc[cohort_trial_cond, cohort_trial_info].reset_index(drop=True))
                cohort_results = pd.concat(cohort_trial_results, axis=1)
                cohort_results.columns = model_results.columns
            cohort_metrics = get_metrics(cohort_results)
            model_name = model if cohort == 'suspected_infection' else f'{cohort}_{model}'
            save_metrics(cohort_metrics, save_path, model_name)


if __name__ == "__main__":
    data_path = '/opt/scratchspace/remoor6/pediatric_sepsis/coi/dataset_24h_filtered.parquet.gzip'
    df = pd.read_parquet(data_path)
    for train_data in ['EG', 'SR']:
        results_path = f"{os.getcwd()}/results/{train_data}/XGB"
        print(f"Performing analysis...")
        for validation in ['internal', 'external']:
            print(f"Performing analysis for {results_path}/{validation}...")
            model_results_path = f'{results_path}/model_results/{validation}'
            metrics_path = f'{results_path}/metrics/{validation}'
            results = {}
            for csv_file in sorted(os.scandir(model_results_path), key=lambda x: x.name):
                model_name = csv_file.name.split('.csv')[0]
                results[model_name] = pd.read_csv(csv_file.path)
            sensitivity_analysis(results, df, metrics_path)