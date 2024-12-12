cat_features = {'pupil_left_reaction': 'Pupil Left Reaction', 
                'pupil_right_reaction': 'Pupil Right Reaction', 
                'abnormal_heart_rate': 'Abnormal Heart Rate', 
                'abnormal_resp_rate': 'Abnormal Respiratory Rate', 
                'abnormal_temp': 'Abnormal Temperature', 
                'abnormal_wbc': 'Abnormal WBC',
                'abnormal_neut_bands': 'Abnormal Band Neutrophils',
                'abnormal_bp_sys': 'Abnormal Systolic Blood Pressure', 
                'abnormal_base_deficit': 'Abnormal Base Deficit', 
                'abnormal_lactate': 'Abnormal Lactate', 
                'cons_spo2_below90': 'Constant SpO2 Below 90', 
                'fio2_above50': 'FiO2 Above 50', 
                'low_platelets': 'Low Platelets', 
                'abnormal_pt': 'Abnormal Prothrombin Time', 
                'abnormal_inr': 'Abnormal INR', 
                'elevated_creat': 'Elevated Creatinine', 
                'abnormal_alt': 'Abnormal ALT', 
                'abnormal_ast': 'Abnormal AST', 
                'on_asthma_meds': 'On Asthma Medications', 
                'on_seizure_meds': 'On Seizure Medications', 
                'on_vasopressors': 'On Vasopressors', 
                'on_antiinf_meds': 'On Anti-Infection Medications', 
                'on_insulin':  'On Insulin', 
                'had_cultures_ordered': 'Had Cultures Ordered', 
                'sepsis_septicemia_diag': 'Sepsis Septicemia Diagnosis', 
                'septic_shock_diag': 'Septic Shock Diagnosis', 
                'sickle_cell_diag': 'Sickle Cell Diagnosis', 
                'dka_diag': 'DKA Diagnosis', 
                'asthmaticus_diag': 'Asthmaticus Diagnosis',
                }

lab_features = {'albumin': 'Albumin',
                'base_deficit': 'Base Deficit',
                'base_excess': 'Base Excess',
                'bicarbonate': 'Bicarbonate',
                'bilirubin_total': 'Total Bilirubin',
                'bp_dias': 'Diastolic Blood Pressure',
                'bp_sys': 'Systolic Blood Pressure',
                'bun': 'BUN',
                'calcium': 'Calcium',
                'calcium_ionized': 'Ionized Calcium',
                'chloride': 'Chloride',
                'co2': 'CO2',
                'coma_scale_total': 'Coma Scale Total',
                'creatinine': 'Creatinine',
                'fio2': 'FiO2',
                'glucose': 'Glucose',
                'hemoglobin': 'Hemoglobin',
                'lactic_acid': 'Lactic Acid',
                'map': 'MAP',
                'o2_flow': 'O2 Flow',
                'pao2_fio2': 'PaO2/FiO2',
                'pco2': 'PCO2',
                'po2': 'PO2',
                'ph': 'pH',
                'platelets': 'Platelets',
                'potassium': 'Postassium',
                'ptt': 'PTT',
                'pulse': 'Pulse',
                'pupil_left_size': 'Pupil Left Size',
                'pupil_right_size': 'Pupil Right Size',
                'resp': 'Respiratory Rate',
                'sodium': 'Sodium',
                'spo2': 'SpO2',
                'temp': 'Temperature',
                'urine': 'Urine',
                'vol_infused': 'Volume Infused',
                'wbc': 'White Blood Cell Count',
                'weight': 'Weight (kg)'
                }

age_psofa_features = {'age_months': 'Age (Months)',
                      'age_years': 'Age (Years)',
                      'resp_psofa': 'Respiratory pSOFA',
                      'coag_psofa': 'Coag pSOFA',
                      'hep_psofa': 'Hepatic pSOFA',
                      'card_psofa': 'Card pSOFA',
                      'neuro_psofa': 'Neuro pSOFA',
                      'renal_psofa': 'Renal pSOFA',
                      'psofa': 'pSOFA'
                      }

# COI features
# education subdomain
ed_features = {'r_ED_EC_stt': 'Early Childhood Education',
               'r_ED_EL_stt': 'Elementary Education',
               'r_ED_SP_stt': 'Secondary and Post-secondary Education',
               'r_ED_ER_stt': 'Education Resources'
               }

# health and environment subdomain
he_features = {'r_HE_EP_stt': 'Pollution',
               'r_HE_HE_stt': 'Healthy Environments',
               'r_HE_SE_stt': 'Safety-related Resources',
               'r_HE_HR_stt': 'Health Resources'
               }

# social and economic subdomain
se_features = {'r_SE_EO_stt': 'Employment',
               'r_SE_ER_stt': 'Economic Resources',
               'r_SE_EI_stt': 'Concentrated Socioeconomic Inequity',
               'r_SE_HQ_stt': 'Housing Resources',
               'r_SE_SR_stt': 'Social Resources',
               'r_SE_WL_stt': 'Wealth',
               }

# subdomain and overall ranks
ranks_features = {'r_ED_stt': 'Education',
                  'r_HE_stt': 'Health and Environment',
                  'r_SE_stt': 'Social and Economic',
                  'r_COI_stt': 'COI'
                  }

coi_features = dict(list(ed_features.items()) + list(he_features.items()) + list(se_features.items()) + list(ranks_features.items()))


# for model development
stats = ['mean', 'median', 'min', 'max', 'std']
lab_stat_features = [f'{x}_{y}' for x in lab_features.keys() for y in stats]
cont_features = lab_stat_features + list(age_psofa_features.keys())
features = {'cat': list(cat_features.keys()),
            'cont': cont_features,
            'coi': list(coi_features.keys())
            }

# for plot titles
stats = ['Mean', 'Median', 'Min', 'Max', 'Std']
lab_stat_plot_features = [f'{x} {y}' for x in lab_features.values() for y in stats]
cont_plot_features = lab_stat_plot_features + list(age_psofa_features.values())
plot_features = {'cat': list(cat_features.values()),
                 'cont': cont_plot_features,
                 'coi': list(coi_features.values())
                 }

plot_colors = {'red': '#e6194b',
               'green': '#3cb44b',
               'blue': '#4363d8',
               'orange': '#f58231',
               'purple': '#911eb4',
               'cyan': '#42d4f4',
               'magenta': '#f032e6',
               'lime': '#bfef45',
               'pink': '#fabed4',
               'teal': '#469990',
               'lavender': '#dcbeff',
               'brown': '#9a6324',
               'beige': '#fffac8',
               'maroon': '#800000',
               'mint': '#aaffc3',
               'olive': '#808000',
               'apricot': '#ffd8b1',
               'navy': '#000075',
               'grey': '#a9a9a9',
               'orange red': '#ff4500',
               'black': '#000000',
               'yellow': '#ffe119'
               }