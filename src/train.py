from .analysis import get_shap_values, save_model_results
from .globalvars import features, plot_features
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from .preprocess import create_pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from tqdm import tqdm
from .utils import get_params


def model_validation(model, train_df, test_df, results_path, args):
    # select to use emr, coi, or emr+coi to develop models
    if args['features'] == 'emr':
        feats = features['cont'] + features['cat']
    elif args['features'] == 'coi':
        feats = features['coi']
    elif args['features'] == 'both':
        feats = features['cont'] + features['cat'] + features['coi']

    label = 'picu_composite'

    # need labels for stratification
    train_labels = train_df[label].values
    train_csns = train_df['csn'].unique()
    test_csns = test_df['csn'].unique()

    # get model hyperparameters
    params = get_params(args['model'])

    # create pipeline
    pipeline = create_pipeline(args['features'])

    val_model_results = {}
    test_model_results = {}
    for trial in tqdm(range(args['trials'])):
        # split train into train / val for cross validation
        train_sample_csns, val_csns = train_test_split(train_csns, train_size=0.7, stratify=train_labels, random_state=trial)

        train_sample_df = train_df.loc[train_df['csn'].isin(train_sample_csns)]
        val_df = train_df.loc[train_df['csn'].isin(val_csns)]

        X_train = train_sample_df[feats]
        y_train = train_sample_df[label].values

        X_val = val_df[feats]
        y_val = val_df[label].values

        X_test = test_df[feats]
        y_test = test_df[label].values
        
        # impute missing data
        X_train_imp = pipeline.fit_transform(X_train)
        X_val_imp = pipeline.transform(X_val)
        X_test_imp = pipeline.transform(X_test)

        # Perform oversampling of training data
        oversample = SMOTE(random_state=trial)
        X_train_imp, y_train = oversample.fit_resample(X_train_imp, y_train)

        # train model find optimal parameters
        cv = StratifiedKFold(shuffle=True, random_state=trial)
        cv_pipeline = RandomizedSearchCV(model, param_distributions=params, scoring='roc_auc', cv=cv, random_state=trial)

        # perform regular training procedure
        cv_pipeline.fit(X_train_imp, y_train)

        # refit optimal model with all train data
        best_pipeline = cv_pipeline.best_estimator_
        best_pipeline.fit(X_train_imp, y_train)

        val_probs = best_pipeline.predict_proba(X_val_imp)[:,1]
        test_probs = best_pipeline.predict_proba(X_test_imp)[:,1]

        val_model_results[f'csn_{trial}'] = val_csns
        val_model_results[f'y_true_{trial}'] = y_val
        val_model_results[f'y_prob_{trial}'] = val_probs

        test_model_results[f'csn_{trial}'] = test_csns
        test_model_results[f'y_true_{trial}'] = y_test
        test_model_results[f'y_prob_{trial}'] = test_probs
    
    # Save test results for plots
    model_name = f"{args['features']}"
    model_results_path = f'{results_path}/model_results'
    val_model_results_path = f'{model_results_path}/internal'
    test_model_results_path = f'{model_results_path}/external'

    save_model_results(val_model_results, val_model_results_path, model_name)
    save_model_results(test_model_results, test_model_results_path, model_name)

    # get shap values
    shap_path = f'{results_path}/shap_data'
    val_shap_path = f'{shap_path}/internal'
    test_shap_path = f'{shap_path}/external'
    if args['features'] == 'emr':
        plot_feats = plot_features['cont'] + plot_features['cat']
    elif args['features'] == 'coi':
        plot_feats = plot_features['coi']
    elif args['features'] == 'both':
        plot_feats = plot_features['cont'] + plot_features['cat'] + plot_features['coi']

    # format data for shap
    shap_train_df = pd.DataFrame(X_train_imp, columns=plot_feats)
    shap_val_df = pd.DataFrame(X_val_imp, columns=plot_feats)
    shap_test_df = pd.DataFrame(X_test_imp, columns=plot_feats)

    # get unscaled data for shap scatter plots
    if args['features'] == 'emr':
        scaler = pipeline.transformers_[0][1].named_steps['scale']
        encoder = pipeline.transformers_[1][1].named_steps['enc']
        cont_feats = len(features['cont'])
        cat_feats = len(features['cat'])

        X_val_cont = scaler.inverse_transform(X_val_imp[:,:cont_feats])
        X_val_cat = encoder.inverse_transform(X_val_imp[:,cont_feats:cont_feats+2])
        X_val_ind = X_val_imp[:,cont_feats+2:]
        
        X_test_cont = scaler.inverse_transform(X_test_imp[:,:cont_feats])
        X_test_cat = encoder.inverse_transform(X_test_imp[:,cont_feats:cont_feats+2])
        X_test_ind = X_test_imp[:,cont_feats+2:]

        X_val_unscaled = np.concatenate([X_val_cont, X_val_cat, X_val_ind], axis=1)
        X_test_unscaled = np.concatenate([X_test_cont, X_test_cat, X_test_ind], axis=1)
    elif args['features'] == 'coi':
        scaler = pipeline.transformers_[0][1].named_steps['scale']
        X_val_unscaled = scaler.inverse_transform(X_val_imp)
        X_test_unscaled = scaler.inverse_transform(X_test_imp)
    elif args['features'] == 'both':
        cont_scaler = pipeline.transformers_[0][1].named_steps['scale']
        encoder = pipeline.transformers_[1][1].named_steps['enc']
        coi_scaler = pipeline.transformers_[2][1].named_steps['scale']
        cont_feats = len(features['cont'])
        cat_feats = len(features['cat'])

        X_val_cont = cont_scaler.inverse_transform(X_val_imp[:,:cont_feats])
        X_val_cat = encoder.inverse_transform(X_val_imp[:,cont_feats:cont_feats+2])
        X_val_ind = X_val_imp[:,cont_feats+2:cont_feats+cat_feats]
        X_val_coi = coi_scaler.inverse_transform(X_val_imp[:,cont_feats+cat_feats:])
        
        X_test_cont = cont_scaler.inverse_transform(X_test_imp[:,:cont_feats])
        X_test_cat = encoder.inverse_transform(X_test_imp[:,cont_feats:cont_feats+2])
        X_test_ind = X_test_imp[:,cont_feats+2:cont_feats+cat_feats]
        X_test_coi = coi_scaler.inverse_transform(X_test_imp[:,cont_feats+cat_feats:])

        X_val_unscaled = np.concatenate([X_val_cont, X_val_cat, X_val_ind, X_val_coi], axis=1)
        X_test_unscaled = np.concatenate([X_test_cont, X_test_cat, X_test_ind, X_test_coi], axis=1)
    
    get_shap_values(best_pipeline, shap_train_df, shap_val_df, X_val_unscaled, val_shap_path, args['features'])
    get_shap_values(best_pipeline, shap_train_df, shap_test_df, X_test_unscaled, test_shap_path, args['features'])