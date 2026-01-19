output "artifact_registry_repo" {
  value = try(google_artifact_registry_repository.repo.name, null)
}

output "gke_cluster_name" {
  value = google_container_cluster.gke.name
}

output "service_external_ip" {
  value = try(data.kubernetes_service_v1.svc[0].status[0].load_balancer[0].ingress[0].ip, null)
}

output "service_external_hostname" {
  value = try(data.kubernetes_service_v1.svc[0].status[0].load_balancer[0].ingress[0].hostname, null)
}

output "raw_bucket"    { 
  value = google_storage_bucket.raw.name 
}
output "models_bucket" { 
  value = google_storage_bucket.models.name 
  }