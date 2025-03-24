import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.custom_exception import CustomException
from src.logger import get_logger
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from config.model_params import *
from scipy.stats import randint
import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTrainer():
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path
        
        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
    
    def load_and_split_data(self):
        try:
            logger.info(f"Loading Train data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading Test data from {self.test_path}")
            test_df = load_data(self.test_path)
            
            logger.info("Splitting the data")
            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']
            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']
            
            logger.info("Data Splitted Successfully")
            
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logger.error(f"Error while loading and splitting data: {e}")
            raise CustomException("Failed to load/split data", e)
    
    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Initializing LGBM Model")
            lgbm_params = {
                "random_state": self.random_search_params['random_state'],
                "num_threads": 1,  # Limit to one thread
                "force_col_wise": True  # Avoid auto-detection overhead
            }

            lgbm = lgb.LGBMClassifier(**lgbm_params)
            
            logger.info("Starting Hyperparameter fine-tuning")
            
            random_search = RandomizedSearchCV(
                estimator=lgbm,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                n_jobs=self.random_search_params['n_jobs'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state'],
                scoring=self.random_search_params['scoring'],
            )
            
            logger.info("Starting Hyperparamater Fine-tuning")
            random_search.fit(X_train, y_train)
            logger.info("Fine-tuning completed")
            
            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_
            
            logger.info(f"Best params are : {best_params}")
            
            return best_lgbm_model
        
        except Exception as e:
            logger.error(f"Error while training model: {e}")
            raise CustomException("Failed to train the model", e)
    
    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating the Model")
            
            y_preds = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_preds)
            precision = precision_score(y_test, y_preds)
            recall = recall_score(y_test, y_preds)
            f1 = f1_score(y_test, y_preds)
            
            logger.info(f"Accuracy Score : {accuracy}")
            logger.info(f"Precision Score : {precision}")
            logger.info(f"Recall Score : {recall}")
            logger.info(f"F1 Score : {f1}")
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        
        except Exception as e:
            logger.error(f"Error while evaluating model: {e}")
            raise CustomException("Failed to evaluate model", e)
    
    def save_model(self, model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            logger.info("Saving the model")
            joblib.dump(model, self.model_output_path)
            logger.info(f"Model saved to {self.model_output_path}")
             
        except Exception as e:
            logger.error(f"Error while saving the model: {e}")
            raise CustomException("Failed to save the model", e)
    
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Running Model Training Script")
                
                logger.info("Starting MLFlow Experimentation")
                
                logger.info("Logging the training and testing dataset to MLFlow")
                
                mlflow.log_artifact(self.train_path, artifact_path='datasets')
                mlflow.log_artifact(self.test_path, artifact_path='datasets')
                
                X_train, X_test, y_train, y_test = self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                self.save_model(best_lgbm_model)
                
                logger.info("Logging the model into MLFlow")
                mlflow.log_artifact(self.model_output_path)
                
                logger.info("Logging Params and Metrics into MLFlow")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)
                
                logger.info("Model Training Successful")
            
        except Exception as e:
            logger.error(f"Error while running model training script: {e}")
            raise CustomException("Failed execute model training script", e)
        
        finally:
            logger.info("Model Training Script completed")

if __name__ == "__main__":
    trainer = ModelTrainer(train_path=PROCESSED_TRAIN_DATA_PATH,
                           test_path=PROCESSED_TEST_DATA_PATH,
                           model_output_path=MODEL_OUTPUT_PATH)
    trainer.run()