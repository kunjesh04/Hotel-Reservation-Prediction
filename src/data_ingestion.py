import os
import pandas as pd
from google.cloud import storage
from config.paths_config import *
from src.logger import get_logger
from utils.common_functions import read_yaml
from src.custom_exception import CustomException
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)

class DataIngestion():
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]
        
        os.makedirs(RAW_DIR, exist_ok=True)
        
        logger.info(f"Data Ingestion Started from {self.file_name} inside {self.bucket_name} bucket.")
        
    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            
            blob.download_to_filename(RAW_FILE_PATH)
            
            logger.info(f"CSV Successfully downloaded to {RAW_FILE_PATH}")
        
        except Exception as e:
            logger.error("Error while downloading CSV file.")
            raise CustomException("Failed to download CSV file.", e)
    
    def split_data(self):
        try:
            logger.info("Starting to split the data.")
            data = pd.read_csv(RAW_FILE_PATH)
            
            train_data, test_data = train_test_split(data, test_size= 1-self.train_test_ratio, random_state= 42)
            
            train_data.to_csv(TRAIN_FILE_PATH)
            test_data.to_csv(TEST_FILE_PATH)
            
            logger.info(f"Train Data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test Data saved to {TEST_FILE_PATH}")
            
        except Exception as e:
            logger.error("Error while splitting the data.")
            raise CustomException("Failed to split the data.", e)
    
    def run(self):
        try:
            logger.info("Starting the Data Ingestion process")
            
            self.download_csv_from_gcp()
            self.split_data()
            
            logger.info("Data ingestion process successful")
        
        except Exception as ce:
            logger.error(f"Custom Exception : {str(ce)}")
        
        finally:
            logger.info("Data Ingestion Completed")

if __name__ == "__main__":
    
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()