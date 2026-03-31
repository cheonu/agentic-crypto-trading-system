pipeline {
    agent any

    environment {
        PROJECT_ID     = credentials('gcp-project-id')
        REGION         = 'us-central1'
        REGISTRY       = "${REGION}-docker.pkg.dev/${PROJECT_ID}/crypto-trader"
        IMAGE_NAME     = 'crypto-trader'
        IMAGE_TAG      = "${env.BUILD_NUMBER}-${env.GIT_COMMIT?.take(7) ?: 'latest'}"
        CLUSTER_NAME   = 'crypto-trader-cluster'
        ZONE           = 'us-central1-a'
    }

    options {
        timeout(time: 30, unit: 'MINUTES')
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Test') {
            steps {
                dir('agentic-crypto-trading-system') {
                    sh 'pip install poetry'
                    sh 'poetry install'
                    sh 'poetry run pytest tests/ -v --tb=short -q'
                }
            }
            post {
                always {
                    junit allowEmptyResults: true,
                          testResults: '**/test-results/*.xml'
                }
            }
        }

        stage('Lint & Type Check') {
            steps {
                dir('agentic-crypto-trading-system') {
                    sh 'poetry run python -m py_compile src/agentic_crypto_trading_system/main.py'
                    sh 'poetry run python -c "from agentic_crypto_trading_system.main import TradingSystem; print(\'Import OK\')"'
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                dir('agentic-crypto-trading-system') {
                    sh """
                        docker build -t ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} .
                        docker tag ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}/${IMAGE_NAME}:latest
                    """
                }
            }
        }

        stage('Push to Registry') {
            steps {
                sh """
                    gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
                    docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
                    docker push ${REGISTRY}/${IMAGE_NAME}:latest
                """
            }
        }

        stage('Deploy to GKE') {
            steps {
                sh """
                    gcloud container clusters get-credentials ${CLUSTER_NAME} \
                        --zone ${ZONE} --project ${PROJECT_ID}
                """
                dir('agentic-crypto-trading-system') {
                    sh """
                        helm upgrade --install crypto-trader ./helm/crypto-trader \
                            --namespace crypto-trader \
                            --create-namespace \
                            --set image.repository=${REGISTRY}/${IMAGE_NAME} \
                            --set image.tag=${IMAGE_TAG} \
                            --wait --timeout 5m
                    """
                }
            }
        }

        stage('Verify Deployment') {
            steps {
                sh """
                    kubectl get pods -n crypto-trader -l app=crypto-trader
                    kubectl rollout status deployment/trader-langgraph-baseline -n crypto-trader --timeout=120s
                    kubectl rollout status deployment/trader-crewai-openai -n crypto-trader --timeout=120s
                    kubectl rollout status deployment/trader-crewai-anthropic -n crypto-trader --timeout=120s
                """
            }
        }
    }

    post {
        success {
            echo "Deployment successful! All 3 trading instances running."
            echo "Monitor: kubectl logs -f -n crypto-trader -l app=crypto-trader"
        }
        failure {
            echo "Pipeline failed. Check logs above."
        }
        always {
            cleanWs()
        }
    }
}
