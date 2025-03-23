pipeline{
    agent any

    stages{
        stage('Cloning Github repo'){
            steps{
                echo 'Cloning Github repo to Jenkins ...'
                checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/kunjesh04/Hotel-Reservation-Prediction.git']])
            }
        }
    }
}