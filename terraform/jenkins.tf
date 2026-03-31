# ─── Jenkins on GCE ───

resource "google_compute_instance" "jenkins" {
  name         = "jenkins-server"
  machine_type = "e2-standard-2"
  zone         = var.zone

  tags = ["jenkins", "http-server", "https-server"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 50
      type  = "pd-balanced"
    }
  }

  network_interface {
    network = "default"
    access_config {
      # Ephemeral public IP
    }
  }

  metadata_startup_script = <<-SCRIPT
    #!/bin/bash
    set -e

    # Install Java (Jenkins requirement)
    apt-get update
    apt-get install -y openjdk-17-jre-headless

    # Install Jenkins
    curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key | tee /usr/share/keyrings/jenkins-keyring.asc > /dev/null
    echo "deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] https://pkg.jenkins.io/debian-stable binary/" | tee /etc/apt/sources.list.d/jenkins.list > /dev/null
    apt-get update
    apt-get install -y jenkins

    # Install Docker
    apt-get install -y docker.io
    usermod -aG docker jenkins

    # Install gcloud CLI
    apt-get install -y apt-transport-https ca-certificates gnupg
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee /etc/apt/sources.list.d/google-cloud-sdk.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
    apt-get update
    apt-get install -y google-cloud-cli google-cloud-cli-gke-gcloud-auth-plugin

    # Install kubectl
    apt-get install -y kubectl

    # Install Helm
    curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

    # Install Python 3.13 + Poetry
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
    apt-get install -y python3.13 python3.13-venv python3-pip
    pip install poetry

    # Start Jenkins
    systemctl enable jenkins
    systemctl start jenkins
  SCRIPT

  service_account {
    email  = google_service_account.jenkins.email
    scopes = ["cloud-platform"]
  }

  labels = {
    app = "jenkins"
    env = "production"
  }
}

# ─── Jenkins Service Account ───

resource "google_service_account" "jenkins" {
  account_id   = "jenkins-sa"
  display_name = "Jenkins CI/CD Service Account"
}

# Jenkins needs to push images, deploy to GKE, access secrets
resource "google_project_iam_member" "jenkins_artifact_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.jenkins.email}"
}

resource "google_project_iam_member" "jenkins_gke_admin" {
  project = var.project_id
  role    = "roles/container.developer"
  member  = "serviceAccount:${google_service_account.jenkins.email}"
}

resource "google_project_iam_member" "jenkins_secrets" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.jenkins.email}"
}

resource "google_project_iam_member" "jenkins_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.jenkins.email}"
}

# ─── Firewall rules ───

resource "google_compute_firewall" "jenkins_http" {
  name    = "allow-jenkins-http"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["jenkins"]
}

resource "google_compute_firewall" "jenkins_https" {
  name    = "allow-jenkins-https"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["443"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["jenkins"]
}

# ─── Output ───

output "jenkins_url" {
  value = "http://${google_compute_instance.jenkins.network_interface[0].access_config[0].nat_ip}:8080"
}

output "jenkins_initial_password_cmd" {
  value = "gcloud compute ssh jenkins-server --zone ${var.zone} --command 'sudo cat /var/lib/jenkins/secrets/initialAdminPassword'"
}
