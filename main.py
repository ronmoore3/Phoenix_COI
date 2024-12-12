import argparse
import os
import pandas as pd
from src.train import model_validation
from src.utils import load_model

'''
Main script for training
'''


def main(args):
    print(f"Model: {args['model']}")
    print(f"Train Data: {args['train_data']}")
    print(f"Features: {args['features']}")
    print(f"Trials: {args['trials']}")

    results_path = f"{os.getcwd()}/results/{args['train_data']}/{args['model']}"

    data_path = '/opt/scratchspace/remoor6/pediatric_sepsis/coi/dataset_24h_filtered.parquet.gzip'
    df = pd.read_parquet(data_path)

    # Select src and tgt data
    if args['train_data'] == 'EG':
        train_site = 'EG PEDIATRIC ICU'
        test_site = 'SR PEDIATRIC ICU'
    else:
        train_site = 'SR PEDIATRIC ICU'
        test_site = 'EG PEDIATRIC ICU'

    train_cond = df['department'] == train_site
    test_cond = df['department'] == test_site

    train_df = df.loc[train_cond].copy()
    test_df = df.loc[test_cond].copy()
    
    # load model
    model = load_model(args['model'])

    print(f"Performing model validation...")
    model_validation(model, train_df, test_df, results_path, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", choices=['EG', 'SR'], default='EG', help="Data set to use as train campus.")
    parser.add_argument("--features", choices=['emr', 'coi', 'both'], default='emr', help="Features to use for model development.")
    parser.add_argument("--model", choices=['LR', 'XGB'], default='XGB', help="Model to train.")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials to use for experiment repitition.")
    args = vars(parser.parse_args())
    main(args)