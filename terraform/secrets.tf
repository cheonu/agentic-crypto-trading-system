# ─── Secret Manager for API keys ───

resource "google_secret_manager_secret" "openai_key" {
  secret_id = "openai-api-key"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "anthropic_key" {
  secret_id = "anthropic-api-key"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "google_ai_key" {
  secret_id = "google-ai-key"
  replication {
    auto {}
  }
}
