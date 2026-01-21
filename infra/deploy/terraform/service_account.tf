resource "google_service_account" "recsys_cloudsql" {
  account_id   = "recsys-cloudsql"
  display_name = "Recsys Cloud SQL access"
}

resource "google_project_iam_member" "cloudsql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.recsys_cloudsql.email}"
}