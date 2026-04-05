pipeline {
    agent {
        kubernetes {
            yaml '''
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: docker
    image: docker:24-dind
    securityContext:
      privileged: true
    resources:
      requests:
        ephemeral-storage: "15Gi"
    volumeMounts:
    - name: docker-storage
      mountPath: /var/lib/docker
    - name: shared
      mountPath: /shared
  - name: gcloud
    image: google/cloud-sdk:slim
    command: ['sleep']
    args: ['infinity']
    volumeMounts:
    - name: shared
      mountPath: /shared
  volumes:
  - name: docker-storage
    emptyDir: {}
  - name: shared
    emptyDir: {}
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

        stage('Build Image') {
            steps {
                container('docker') {
                    sh """
                        docker build --platform linux/amd64 \
                            -t ${FULL_IMAGE} \
                            -t ${LATEST_IMAGE} \
                            .
                    """
                }
            }
        }

        stage('Auth to Artifact Registry') {
            steps {
                container('gcloud') {
                    withCredentials([file(credentialsId: 'gcp-service-account', variable: 'GCP_KEY')]) {
                        sh """
                            gcloud auth activate-service-account --key-file=\$GCP_KEY
                            gcloud auth print-access-token | tr -d '\\n' > /shared/gcp-token
                        """
                    }
                }
            }
        }

        stage('Push to Artifact Registry') {
            steps {
                container('docker') {
                    sh """
                        TOKEN=\$(cat /shared/gcp-token)
                        echo "\$TOKEN" | docker login -u oauth2accesstoken --password-stdin https://${REGION}-docker.pkg.dev
                        docker push ${FULL_IMAGE}
                        docker push ${LATEST_IMAGE}
                    """
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
