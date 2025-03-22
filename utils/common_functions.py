import os
import pandas
from src.custom_exception import CustomException
from src.logger import get_logger
import yaml

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File is not in the given path")
        
        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(file_path)
            logger.info("Successfully read the YAML file")
            return config
    except Exception as e:
        logger.error("Error while reading the YAML file")
        raise CustomException("Failed to read YAML file", e)