pipeline {
    agent {
        kubernetes {
            yaml '''
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: gcloud
    image: google/cloud-sdk:slim
    command: ['sleep']
    args: ['infinity']
  volumes: []
'''
        }
    }

    environment {
        PROJECT_ID     = 'project-e4ad9f18-82a4-4e98-ae4'
        REGION         = 'europe-west1'
        REGISTRY       = "${REGION}-docker.pkg.dev/${PROJECT_ID}/crypto-trader"
        IMAGE_NAME     = 'crypto-trader'
        IMAGE_TAG      = "${env.BUILD_NUMBER}-${env.GIT_COMMIT?.take(7) ?: 'latest'}"
        FULL_IMAGE     = "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
        LATEST_IMAGE   = "${REGISTRY}/${IMAGE_NAME}:latest"
        K8S_NAMESPACE  = 'crypto-trader'
        GKE_CLUSTER    = 'crypto-trader-eu'
        GKE_ZONE       = 'europe-west1-b'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build and Push Image') {
            steps {
                container('gcloud') {
                    withCredentials([file(credentialsId: 'gcp-service-account', variable: 'GCP_KEY')]) {
                        sh """
                            gcloud auth activate-service-account --key-file=\$GCP_KEY
                            
                            # Submit build asynchronously
                            BUILD_ID=\$(gcloud builds submit \
                                --project ${PROJECT_ID} \
                                --region ${REGION} \
                                --tag ${FULL_IMAGE} \
                                --timeout=1800s \
                                --async \
                                --format='value(id)' \
                                .)
                            
                            echo "Cloud Build started: \$BUILD_ID"
                            
                            # Wait for build to complete
                            while true; do
                                STATUS=\$(gcloud builds describe \$BUILD_ID \
                                    --project ${PROJECT_ID} \
                                    --region ${REGION} \
                                    --format='value(status)')
                                echo "Build status: \$STATUS"
                                if [ "\$STATUS" = "SUCCESS" ]; then
                                    break
                                elif [ "\$STATUS" = "FAILURE" ] || [ "\$STATUS" = "TIMEOUT" ] || [ "\$STATUS" = "CANCELLED" ]; then
                                    echo "Build failed with status: \$STATUS"
                                    exit 1
                                fi
                                sleep 30
                            done
                            
                            # Tag as latest
                            gcloud artifacts docker tags add ${FULL_IMAGE} ${LATEST_IMAGE} 2>/dev/null || true
                        """
                    }
                }
            }
        }

        stage('Deploy to K8s') {
            steps {
                container('gcloud') {
                    withCredentials([file(credentialsId: 'gcp-service-account', variable: 'GCP_KEY')]) {
                        sh """
                            apt-get update -qq && apt-get install -y -qq kubectl google-cloud-cli-gke-gcloud-auth-plugin > /dev/null 2>&1
                            export USE_GKE_GCLOUD_AUTH_PLUGIN=True
                            gcloud auth activate-service-account --key-file=\$GCP_KEY
                            gcloud container clusters get-credentials ${GKE_CLUSTER} --zone ${GKE_ZONE} --project ${PROJECT_ID}

                            # Install Helm
                            curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

                            # Deploy with Helm
                            helm upgrade crypto-trader helm/crypto-trader \
                                --namespace ${K8S_NAMESPACE} \
                                --set image.repository=${REGISTRY}/${IMAGE_NAME} \
                                --set image.tag=${IMAGE_TAG} \
                                --set testnet.enabled=true \
                                --wait --timeout 300s
                        """
                    }
                }
            }
        }
    }

    post {
        success {
            echo "Deployed ${FULL_IMAGE} to crypto-trader namespace successfully"
        }
        failure {
            echo "Pipeline failed. Check logs above."
        }
    }
}
