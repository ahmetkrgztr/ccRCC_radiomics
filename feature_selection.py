import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import logging
from imblearn.over_sampling import ADASYN,SMOTE, SVMSMOTE
from tools.data_processing import FeatureSelector
from tqdm import tqdm
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model     import LogisticRegression
from tools.stability import getStability
import os
from collections import Counter



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureSelection:
    """
    Select most important features from feature set
    Args:
        df_path: path to Dataframe which includes features
        roin_name: name of region of interest
        n_feature: Number of features wanted to select
        n_cv: Number of fold for cross validation
        num_iter: number of repeatation of feature selection


    """
    ups_methods = ["NONUPSAMPLED", "ADASYN", "SMOTE", "SVMSMOTE"]

    def __init__(self, df_path: str, roi_name: str, n_feature: int, n_cv: int, num_iter=100):
        self.org_df = pd.read_csv(df_path)
        self.roi_name = roi_name
        self.out_feature_num = n_feature
        self.n_cv = n_cv
        self.num_iter = num_iter
        
        logger.info(f"Number of features at the beginning {len(self.org_df.columns)} ")
        logger.info(f"{n_feature} features will be selected")
        logger.info(f"{n_cv} folds will be created for cross validation")
    
    def preprocess(self, features):
        stand_features = features.copy()
        stand_features_columns = features.columns
        scale = StandardScaler().fit(stand_features)
        stand_features = scale.transform(stand_features)

        stand_features = pd.DataFrame(stand_features)
        stand_features.columns = stand_features_columns
        return stand_features

    def remove_constants(self,X_train, y_train):
        # Define steps
        step1 = {'Constant Features': {'frac_constant_values': 0.95}}
        steps = [step1]
        fs = FeatureSelector()
        fs.fit(X_train, y_train, steps)
        logger.info(f"# features before selection: {X_train.shape[1]}")
        X_selected = fs.transform(X_train)
        logger.info(f"# features after removing constants: {X_selected.shape[1]}")
        return X_selected

    def remove_correlated(self,X_train, y_train):
        step1 = {'Correlated Features': {'correlation_threshold': 0.95}}
        steps = [step1]
        fs = FeatureSelector()
        fs.fit(X_train, y_train, steps)
        logger.info(f"# features before selection:  {X_train.shape[1]}")
        X_selected = fs.transform(X_train)
        logger.info(f"# features after removing higly correlateds: {X_selected.shape[1]}")
        return X_selected

    def select_important_fatures(self, X_train, y_train):
        
        feature_names = []
        for i in tqdm(range(self.num_iter)):  
            sel = SelectFromModel(LinearSVC(C=0.5, penalty="l1", dual=False,max_iter=4000))
            sel.fit(X_train, y_train)
            selected_feat= X_train.columns[(sel.get_support())]
            X_sel_rf = X_train[selected_feat]

            cancer_back = SFS(LogisticRegression(max_iter = 1500), k_features=self.out_feature_num, forward=False, floating=False, scoring = 'accuracy',cv=4, n_jobs=-1)
            cancer_back.fit(X_sel_rf, y_train)
            feature_names.append(cancer_back.k_feature_names_)
        return feature_names

    def calculate_stability(self,feature_names):
        stability_table = pd.DataFrame(np.zeros((self.num_iter,self.org_df.shape[1])))
        stability_table.columns = self.org_df.columns
        for i in range(self.num_iter):
            stability_table.iloc[i][list(feature_names[i])] = 1
        stab=getStability(np.array(stability_table))
        return stab, stability_table
    
    def save_selected_features(self,X, y, stability_table,fold=0, ups_method = "SMOTE", stab = 1, tr_ts = "train"):
        selected = X[list(list(stability_table.sum().sort_values()[-self.out_feature_num:].index))].copy()
        selected.index = X.index
        selected["label"] = y.copy()
        path = os.path.join("selected_features",self.roi_name,f"FOLD{fold}", ups_method, f"{tr_ts}_stab{stab}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        selected.to_csv(path,index=False)
        logger.info(f"X_train and y_train shapes {selected.shape} {y.shape}")
        logger.info(f"Saving df to {path}")


    def select_features(self,):
        grades = pd.DataFrame(self.org_df.grade.copy())
        features = self.org_df.copy().drop("grade",axis=1)
        
        skf = StratifiedKFold(n_splits=5)
        skf.get_n_splits(features, grades)

        logger.info(f"Selecting features of {self.roi_name}")

        for idx, (train_ix, test_ix) in enumerate(skf.split(features,grades)):

            for ups_method in self.ups_methods:
                logger.info(f"ROI {self.roi_name}")
                logger.info(f"FOLD {idx}")

                X_train, X_test = features.iloc[train_ix], features.iloc[test_ix]
                y_train, y_test = grades.iloc[train_ix], grades.iloc[test_ix]
                y_train, y_test = np.array(y_train.grade), np.array(y_test.grade)

                logger.info(f"Before upsampling X_train {sorted(Counter(y_train).items())} ")
                logger.info(f"Before upsampling X_train {sorted(Counter(y_test).items())} ")
                logger.info(f"Shape X_Train and y_train at tha beg. of fold  {X_train.shape, y_train.shape}")

                
                if ups_method == "ADASYN":
                    X_train, y_train = ADASYN(sampling_strategy='auto',random_state=41,n_neighbors=5, n_jobs=1).fit_resample(X_train, y_train)
                elif ups_method == "SMOTE":
                    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
                elif ups_method == "SVMSMOTE":
                    X_train, y_train = SVMSMOTE(random_state=400).fit_resample(X_train, y_train)

                X_train, X_test = self.preprocess(X_train), self.preprocess(X_test)

                logger.info(f"UPSAMPLING {ups_method}")
                logger.info(f"Shape X_Train and y_train after upsample {X_train.shape, y_train.shape}")
                
                X_train = self.remove_constants(X_train, y_train)
                logger.info(f"Shape X_Train and y_train after remove_constants { X_train.shape, y_train.shape}")
                X_train = self.remove_correlated(X_train, y_train)
                logger.info(f"Shape X_Train and y_train after CCA {X_train.shape, y_train.shape}")
                feature_names = self.select_important_fatures(X_train, y_train)
                stab, stability_table = self.calculate_stability(feature_names)
                self.save_selected_features(X_train, y_train, stability_table,fold=idx, ups_method = ups_method, stab = stab, tr_ts = "train")
                self.save_selected_features(X_test, y_test, stability_table,fold=idx, ups_method = ups_method, stab = stab, tr_ts="test")


        