provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

data "google_client_config" "default" {}

provider "kubernetes" {
  host                   = try("https://${google_container_cluster.gke.endpoint}", null)
  token                  = try(data.google_client_config.default.access_token, null)
  cluster_ca_certificate = try(base64decode(google_container_cluster.gke.master_auth[0].cluster_ca_certificate), null)
}