import os
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor():
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        
        self.config = read_yaml(config_path)
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)
        
    def preprocess_data(self, df:pd.DataFrame):
        try:
            logger.info("Starting Data Processing")
            
            logger.info("Dropping the columns")
            
            df.drop(columns=['Unnamed: 0', 'Booking_ID'], inplace=True)
            df.drop_duplicates(inplace=True)
            
            categorical_cols = self.config["data_processing"]["categorical_columns"]
            numerical_cols = self.config["data_processing"]["numerical_columns"]
            
            logger.info("Fetched categorical and numerical columns")
            
            encoder = LabelEncoder()
            mappings = {}
            
            for col in categorical_cols:
                df[col] = encoder.fit_transform(df[col])
                mappings[col] = {label:code for label, code in zip(encoder.classes_, encoder.transform(encoder.classes_))}
            
            logger.info("Categorical Columns are Encoded with labels")
            
            logger.info("Handling Skewness of data")
            
            skewness_threshold = self.config["data_processing"]["skewness_threshold"]
            
            skewness = df[numerical_cols].apply(lambda x: x.skew())
            
            for col in skewness[skewness>skewness_threshold].index:
                    df[col] = np.log1p(df[col])
            
            logger.info("Returning the df")
            return df
                        
        except Exception as e:
            logger.error(f"Error during preprocessing data: {e}")
            raise CustomException("Failed to preprocess data", e)
    
    def balance_data(self, df):
        try:
            logger.info("Balancing the imbalanced data")
            
            X = df.drop(columns='booking_status')
            y = df['booking_status']
        
            smote = SMOTE(random_state=42)
            
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df["booking_status"] = y_resampled
            
            logger.info("Data balanced successfully")
            return balanced_df
        
        except Exception as e:
            logger.info(f"Error during balancing data {e}")
            raise CustomException("Failed to Balance Data", e)
    
    def select_features(self, df):
        try:
            logger.info("Selecting Features")
            
            X = df.drop(columns='booking_status')
            y = df['booking_status']
            
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importance
            })
            
            top_features_df = feature_importance_df.sort_values(by='importance', ascending=False)
            
            num_of_features_to_select = self.config["data_processing"]["no_of_features"]
            
            top_n_features = top_features_df["feature"].head(num_of_features_to_select).values
            
            logger.info(f"Top features are : {top_n_features}")
            
            top_n_df = df[top_n_features.tolist() + ["booking_status"]]
            
            logger.info("Feature Selection completed successfully")
            return top_n_df
        
        except Exception as e:
            logger.info(f"Error during feature selection {e}")
            raise CustomException("Failed while feature selection", e)
    
    def save_data(self, df, file_path):
        try:
            logger.info("Saving data in preprocessed directory")
            
            df.to_csv(file_path, index=False)
            logger.info(f"Saved data at {file_path}")
        
        except Exception as e:
            logger.info(f"Error during saving data:  {e}")
            raise CustomException("Failed to save data ", e)
    
    def process(self):
        try:
            logger.info("Starting Data Preprocessing")
            logger.info(f"Loading Data from {RAW_DIR}")
            
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)
            
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)
            
            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)
            
            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]
            
            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)
            
            logger.info("Data preprocessing successful")
        
        except Exception as ce:
            logger.error(f"Custom Exception while preprocessing data: {str(ce)}")
        
        finally:
            logger.info("Data Preprocessing Completed")

if __name__ == "__main__":
    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()