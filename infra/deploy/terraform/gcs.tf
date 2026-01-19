locals {
  raw_bucket_name    = "${var.project_id}-recsys-raw"
  models_bucket_name = "${var.project_id}-recsys-models"
}

resource "google_storage_bucket" "raw" {
  name                        = local.raw_bucket_name
  location                    = var.region
  project                     = var.project_id
  uniform_bucket_level_access = true
  force_destroy               = true

  versioning { enabled = false }
}

resource "google_storage_bucket" "models" {
  name                        = local.models_bucket_name
  location                    = var.region
  project                     = var.project_id
  uniform_bucket_level_access = true
  force_destroy               = true

  versioning { enabled = true }
}

resource "google_storage_bucket_object" "users" {
  bucket = google_storage_bucket.raw.name
  name   = "ml-1m/users.dat"
  source = "${path.module}/../../../data/raw/ml-1m/users.dat"
}

resource "google_storage_bucket_object" "movies" {
  bucket = google_storage_bucket.raw.name
  name   = "ml-1m/movies.dat"
  source = "${path.module}/../../../data/raw/ml-1m/movies.dat"
}

resource "google_storage_bucket_object" "ratings" {
  bucket = google_storage_bucket.raw.name
  name   = "ml-1m/ratings.dat"
  source = "${path.module}/../../../data/raw/ml-1m/ratings.dat"
}