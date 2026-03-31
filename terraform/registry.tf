# ─── Artifact Registry for Docker images ───

resource "google_artifact_registry_repository" "docker" {
  location      = var.region
  repository_id = "crypto-trader"
  format        = "DOCKER"
  description   = "Docker images for crypto trading system"
}

# ─── Service Account for workloads ───

resource "google_service_account" "trader" {
  account_id   = "crypto-trader-sa"
  display_name = "Crypto Trader Service Account"
}

resource "google_project_iam_member" "trader_logging" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.trader.email}"
}

resource "google_project_iam_member" "trader_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.trader.email}"
}

resource "google_project_iam_member" "trader_secrets" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.trader.email}"
}

# Workload Identity binding
resource "google_service_account_iam_member" "workload_identity" {
  service_account_id = google_service_account.trader.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[crypto-trader/crypto-trader-sa]"
}
