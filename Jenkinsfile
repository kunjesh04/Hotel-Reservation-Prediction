pipeline{
    agent any

    stages{
        stage('Cloning Github repo'){
            steps{
                echo 'Cloning Github repo to Jenkins ...'
                git branch: 'main', credentialsId: 'github-token', url: 'https://github.com/kunjesh04/Hotel-Reservation-Prediction.git'
            }
        }
    }
}