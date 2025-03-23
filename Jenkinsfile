pipeline{
    agent any

    environment {
        VENV_DIR = 'venv'
    }

    stages{
        stage('Cloning Github repo'){
            steps{
                echo 'Cloning Github repo to Jenkins ...'
                checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/kunjesh04/Hotel-Reservation-Prediction.git']])
            }
        }
        
        stage('Setting up venv'){
            steps{
                echo 'Setting virtual environment and installing dependencies ...'
                sh '''
                python -m venv ${VENV_DIR}
                . ${VENV_DIR}/bin/activate
                pip install --upgrade pip
                pip install -e .
                '''
            }
        }
    }
}