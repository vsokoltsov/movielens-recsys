resource "google_service_account_iam_member" "workload_identity_user" {
  service_account_id = google_service_account.recsys_cloudsql.name
  role               = "roles/iam.workloadIdentityUser"

  member = "serviceAccount:${var.project_id}.svc.id.goog[${var.k8s_namespace}/${local.ksa_name}]"
}

resource "google_storage_bucket_iam_member" "models_object_viewer" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.recsys_cloudsql.email}"
}