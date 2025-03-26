# Hotel Reservation Prediction System

🚀 **End-to-End MLOps Implementation** 🚀

This project is a **Hotel Reservation Prediction System** that automates data ingestion, preprocessing, model training, and deployment using MLOps best practices. The system predicts whether a hotel booking will be canceled based on booking patterns and customer details. 

---

## 📌 Features
- **Automated Data Pipeline**: Fetches, processes, and stores data from Google Cloud Storage.
- **Feature Engineering & Model Training**: Uses LightGBM and tracks experiments with MLFlow.
- **Full-Stack Deployment**: Flask-based UI, Dockerized, and deployed on Google Cloud Run.
- **CI/CD Pipeline**: Implemented with Jenkins, GitHub, and Google Cloud Platform.

---

## 📂 Project Structure
```bash
├── src/                  # Source code
├── artifacts/            # Stores processed data & models
├── notebook/             # Jupyter notebooks for EDA & feature selection
├── config/               # Configuration files
│   ├── config.yaml       # General configurations
│   ├── paths_config.py   # Paths for artifacts & files
│   ├── model_params.py   # Model hyperparameters
├── utils/                # Utility functions
│   ├── common_functions.py  # Reusable functions
├── pipeline/             # Training pipeline
│   ├── training_pipeline.py # Automates the full pipeline
├── templates/            # HTML files for UI
├── static/               # CSS, JS, and other assets
├── setup.py              # Project setup script
├── requirements.txt      # List of dependencies
├── logger.py             # Logging utility
├── exception.py          # Custom exception handling
└── README.md             # Project documentation
```

---

## 🔧 Project Setup
### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Set Up Google Cloud Platform (GCP)
- Create a **GCP bucket**
- Create a **Service Account** with the following roles:
  - **Storage Admin**
  - **Storage Object Viewer**
- Download the service account JSON key and set the credentials:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your-key.json"
```

---

## 🛠️ Key Components
### **1️⃣ Data Ingestion**
```python
class DataIngestion():
    def __init__(self, config):
        ...
    def download_csv_from_gcp(self):
        ...
    def split_data(self):
        ...
    def run(self):
        ...
```

### **2️⃣ Data Preprocessing**
```python
class DataPreprocessing():
    def __init__(self, config):
        ...
    def preprocess_data(self, df: pd.DataFrame):
        ...
    def balance_data(self, df):
        ...
    def select_features(self, df):
        ...
    def save_data(self, df):
        ...
    def process(self):
        ...
```

### **3️⃣ Model Training**
```python
class ModelTrainer():
    def __init__(self, train_path, test_path, model_output_path):
        ...
    def load_and_split_data(self):
        ...
    def train_lgbm(self, X_train, y_train):
        ...
    def evaluate_model(self, model, X_test, y_test):
        ...
    def save_model(self, model):
        ...
    def run(self):
        ...
```

### **4️⃣ MLFlow Experiment Tracking**
- Tracks hyperparameters, metrics, and model performance.

### **5️⃣ Full Training Pipeline**
```python
if __name__ == "__main__":
    ### Data Ingestion ###
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
    
    ### Data Preprocessing ###
    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()
    
    ### Model Training ###
    trainer = ModelTrainer(train_path=PROCESSED_TRAIN_DATA_PATH,
                           test_path=PROCESSED_TEST_DATA_PATH,
                           model_output_path=MODEL_OUTPUT_PATH)
    trainer.run()
```

---

## 🌍 Deployment & CI/CD Pipeline
### **CI/CD Workflow**
1️⃣ **Set Up Jenkins** (inside a Docker container)
2️⃣ **Integrate GitHub** for automated builds
3️⃣ **Build Docker Image** & push to **Google Container Registry (GCR)**
4️⃣ **Deploy to Google Cloud Run** for serverless execution

### **Commands for CI/CD**
```bash
# Build Docker image
docker build -t hotel-reservation-prediction .

# Push to GCR
docker tag hotel-reservation-prediction gcr.io/YOUR_PROJECT_ID/hotel-reservation

docker push gcr.io/YOUR_PROJECT_ID/hotel-reservation

# Deploy to Cloud Run
gcloud run deploy hotel-reservation \
    --image gcr.io/YOUR_PROJECT_ID/hotel-reservation \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

---

## 🎨 Frontend (Flask UI)
- **Templates:** HTML, CSS, and Bootstrap
- **Designed UI with inspiration from Claude**
- **User can enter booking details & get prediction**

---

## 📌 Links
- **GitHub Repository**: [Hotel Reservation Prediction](https://github.com/kunjesh04/Hotel-Reservation-Prediction)
- **Live Demo**: [Try it here](https://hotel-res-pred-1081285062186.us-central1.run.app/)

---

## 🚀 Future Improvements
- Implement **real-time prediction API**
- Add **more ML models** & ensemble learning
- Enhance UI for **better user experience**

---
