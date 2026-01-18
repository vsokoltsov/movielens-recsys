resource "google_artifact_registry_repository" "repo" {
  depends_on = [google_project_service.artifactregistry]

  project       = var.project_id
  location      = var.region
  repository_id = "recsys"
  description   = "Docker images for recsys"
  format        = "DOCKER"
}