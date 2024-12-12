from .globalvars import features
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline  import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from .scaler import COIScaler


def create_pipeline(feature_type):
    """
    Normalize continuous data and encode categorical data
    """
    # create feature pipeline
    pipelines = []

    # create pipeline for continuous emr features
    if feature_type in ['emr', 'both']:
        cont_imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        cont_pipeline = Pipeline(steps=[('impute', cont_imputer),
                                        ('scale', scaler)
                                        ]
                                        )
        pipelines.append(('cont', cont_pipeline, features['cont']))

        # create pipeline for pupil reactions
        pupil_features = ['pupil_left_reaction', 'pupil_right_reaction']
        pupil_left_categories = ['Not assessed', 'Unable to Assess', 'Non-reactive', 'Reactive']
        pupil_right_categories = ['Not assessed', 'Unable to Assess', 'Non-reactive', 'Reactive']
        pupil_categories = [pupil_left_categories, pupil_right_categories]
        cat_imputer = SimpleImputer(strategy='constant', missing_values=None, fill_value='Not assessed')
        ord_enc = OrdinalEncoder(categories=pupil_categories, handle_unknown='use_encoded_value', unknown_value=np.nan)
        cat_pipeline = Pipeline(steps=[('impute', cat_imputer),
                                       ('enc', ord_enc)
                                       ]
                                )
        pipelines.append(("cat", cat_pipeline, pupil_features))

    # create pipeline for coi features
    if feature_type in ['coi', 'both']:
        coi_scaler = COIScaler()
        coi_pipeline = Pipeline(steps=[('scale', coi_scaler)]
                               )
        pipelines.append(('coi', coi_pipeline, features['coi']))
    
    # pass through for indicator features
    if feature_type == 'coi':
        col_transformer = ColumnTransformer(transformers=pipelines,
                                            remainder='drop'
                                            )
    else:
        col_transformer = ColumnTransformer(transformers=pipelines,
                                             remainder='passthrough'
                                             )
    
    return col_transformer