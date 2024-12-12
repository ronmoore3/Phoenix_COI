from .globalvars import features
import numpy as np
import re
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.metrics import confusion_matrix

# models
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier


def find_sensitivity_threshold(y_true, y_prob, sensitivity=0.85):
    '''
    Get sensitivity value closest to desired sensitivity
    '''
    _, tpr, thresh = roc_curve(y_true, y_prob)
    dist = tpr - sensitivity
    dist = np.where(dist < 0, np.inf, dist)
    optimal_idx = dist.argmin()
    optimal_thresh = thresh[optimal_idx]
    return optimal_thresh


def get_predictions(probs, y=None, sensitivity=None):
    '''
    Gather predictions
    '''
    # find probability threshold if certain sensitivity desired
    if y is not None and sensitivity is not None:
        prob_thresh = find_sensitivity_threshold(y, probs, sensitivity=sensitivity)
    else:
        prob_thresh = 0.5
    preds = (probs >= prob_thresh).astype(int)

    return preds


def expected_calibration_error(y_prob, y_true, bins=5):
    # uniform binning approach with number of bins
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(y_prob, axis=1)
    # get predictions from confidences (positional in this case)
    y_pred = np.argmax(y_prob, axis=1)

    # get a boolean list of correct/false predictions
    accuracies = y_pred==y_true

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

    return ece


def get_features(feature_list, plot=False):
    """
    Get correct feature names in feature list
    """
    final_features = {'cat': [],
                      'cont': []
                      }
    
    for feature in feature_list:
        # check whether variable is categorical or continuous
        if feature in list(features['cat'].keys()):
            final_features['cat'].append(features['cat'][feature] if plot else feature)
        else:
            # for continuous, check if feature contains min, max, mean, median, std
            if re.search("_min|_max|_mean|_median|_std", feature):
                feature_base = re.split("_min|_max|_mean|_median|_std", feature)[0]
                feature_appendix = feature.replace(feature_base, '').replace('_', ' ')
            else:
                feature_base = feature
                feature_appendix = None
            # check if base feature is continuous
            if feature_base in list(features['cont'].keys()):
                if feature_appendix is not None:
                    plot_feature = features['cont'][feature_base] + feature_appendix.title()
                else:
                    plot_feature = features['cont'][feature]
                final_features['cont'].append(plot_feature if plot else feature)

    # move pupil left reaction and pupil right reaction to front of categorical list if present
    if plot:
        if 'Pupil Right Reaction' in final_features['cat']:
            final_features['cat'].remove('Pupil Right Reaction')
            final_features['cat'].insert(0, 'Pupil Right Reaction')
        if 'Pupil Left Reaction' in final_features['cat']:
            final_features['cat'].remove('Pupil Left Reaction')
            final_features['cat'].insert(0, 'Pupil Left Reaction')
    else:
        if 'pupil_right_reaction' in final_features['cat']:
            final_features['cat'].remove('pupil_right_reaction')
            final_features['cat'].insert(0, 'pupil_right_reaction')
        if 'pupil_left_reaction' in final_features['cat']:
            final_features['cat'].remove('pupil_left_reaction')
            final_features['cat'].insert(0, 'pupil_left_reaction')

    return final_features


def get_params(model_name):
    if model_name == 'LR':
        params = {'C': sp_uniform(1, 101)}
    elif model_name == 'RF':
        params = {'n_estimators': sp_randint(20, 101),
                  'max_depth': sp_randint(3, 6)
                  }
    else:
        params = {'clf__learning_rate': [1e-5, 1e-4, 1e-3, 1e-2],
                  'clf__n_estimators': sp_randint(20, 101),
                  'clf__max_depth': sp_randint(3, 6),
                  'clf__scale_pos_weight': sp_uniform(1, 101)
                  }

    return params


def load_model(model_name):
    """
    Load model to be used for training
    """
    if model_name == 'LightGBM':
        model = LGBMClassifier(verbosity=-1, random_state=0)
    elif model_name == 'LR':
        model = LogisticRegression(max_iter=10000, random_state=0)
    elif model_name == 'RF':
        model = RandomForestClassifier(random_state=0)
    elif model_name == 'XGB':
        model = XGBClassifier(verbosity=0, tree_method='gpu_hist', random_state=0)
    
    return model