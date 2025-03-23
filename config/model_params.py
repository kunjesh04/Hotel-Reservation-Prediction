from scipy.stats import randint, uniform

LIGHTGBM_PARAMS = {
    'n_estimators': randint(1, 100),
    'max_depth': randint(5, 50),
    'learning_rate': uniform(0.02, 0.03),
    'num_leaves': randint(20, 100),
    'boosting_type': ['gbdt', 'dart', 'goss']
}

RANDOM_SEARCH_PARAMS = {
    'n_iter': 4,
    'cv': 3,
    'n_jobs': -1,
    'verbose': 2,
    'random_state': 42,
    'scoring': 'accuracy'
}