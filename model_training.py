import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import lightgbm as lgb
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTraining:
    """
    Train models
    Args:
        train_df_path: str Path to train_df
        test_df_path: str Path to test_df
        roi_name: str Name of roi
        fold: int Fold number
        ups_method: str Upsampling method
    """

    def __init__(self, train_df_path: str, 
                test_df_path: str,
                roi_name: str,
                fold: int,
                ups_method: str,
                output_folder: str
                ):

        self.roi_name = roi_name
        self.fold = fold
        self.ups_method = ups_method
        self.output_folder = output_folder

        train_data = pd.read_csv(train_df_path)
        test_data = pd.read_csv(test_df_path)

        self.y_train = train_data["label"]
        self.X_train = train_data.drop(["label"], axis=1)
        self.y_test = test_data["label"]
        self.X_test = test_data.drop(["label"], axis=1)
    
    def train_lgbm(self,):
        logger.info("Training LightGBM")

        SEARCH_PARAMS = {'learning_rate': 0.4,
                'max_depth': 15,
                'num_leaves': 32,
                'feature_fraction': 0.8,
                'subsample': 0.2}

        FIXED_PARAMS={'objective': 'binary',
                    'metric': 'auc',
                    'is_unbalance':False,
                    'bagging_freq':5,
                    'boosting':'dart',
                    'num_boost_round':200,
                    'early_stopping_rounds':300}

        
        X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size=0.2,stratify=self.y_train,random_state=50)
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        params = {'metric':FIXED_PARAMS['metric'],
                'objective':FIXED_PARAMS['objective'],
                **SEARCH_PARAMS}

        model = lgb.train(params, train_data,                     
                        valid_sets=[valid_data],
                        num_boost_round=FIXED_PARAMS['num_boost_round'],
                        early_stopping_rounds=FIXED_PARAMS['early_stopping_rounds'],
                        valid_names=['valid'])
        score = model.best_score['valid']['auc']
        path = os.path.join(self.output_folder, self.roi_name, f"FOLD{self.fold }", self.ups_method,'lgbm.pkl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)

    def train_rf(self,):
        logger.info("Training Random Forest")
        rf = RandomForestClassifier()
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 100)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt', 'log2']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 22)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10,20]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 500, cv = 4, scoring = 'roc_auc', verbose=1, random_state=42, n_jobs = -1)
        # Fit the random search model
        rf_random.fit(self.X_train, self.y_train)
        model = rf_random.best_estimator_
        path = os.path.join(self.output_folder, self.roi_name, f"FOLD{self.fold }", self.ups_method,'rf.pkl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
    
    def train_subspaceKNN(self,):
        logger.info("Training subspace KNN")
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        k_range= list(range(1,15))
        # Create the random grid
        random_grid = {'max_features': [10,12,15],
                    'max_samples': [10,20,30,40,50,60,70,80,90,100],
                    'n_estimators': [50,100,150,200,250,300,400,500],
                    #'n_neighbors': k_range,
                    #'max_depth': max_depth,
                    #'min_samples_split': min_samples_split,
                    #'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
        knn = KNeighborsClassifier(n_neighbors=5)
        knn_classifier = BaggingClassifier(base_estimator = knn ,
                                        max_samples = 10,
                                        n_estimators = 100)
        knn_random = RandomizedSearchCV(estimator = knn_classifier, 
                                param_distributions = random_grid, 
                                n_iter = 1000, 
                                cv = 4, 
                                verbose=2, 
                                random_state=42, 
                                n_jobs = -1,
                               scoring='roc_auc')
        # Fit the random search model
        knn_random.fit(self.X_train, self.y_train)
        model = knn_random.best_estimator_
        path = os.path.join(self.output_folder, self.roi_name, f"FOLD{self.fold }", self.ups_method,'subspaceKNN.pkl')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)

    def train_all(self,):
        self.train_lgbm()
        self.train_rf()
        self.train_subspaceKNN()