output "cluster_name" {
  value = "farm-ai-cluster (existing, not managed by Terraform)"
}

output "registry_url" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}"
}

output "service_account_email" {
  value = google_service_account.trader.email
}
