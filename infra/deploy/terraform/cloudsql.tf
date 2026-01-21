resource "google_sql_database_instance" "postgres" {
  name             = "${var.project_id}-recsys-postgres"
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier = "db-custom-1-3840" # пример

    ip_configuration {
      ipv4_enabled = true
    }
  }

  deletion_protection = false
}

resource "google_sql_database" "db" {
  name     = "recsys"
  instance = google_sql_database_instance.postgres.name
}

resource "random_password" "db_password" {
  length  = 24
  special = true
}

resource "google_sql_user" "app" {
  name     = "recsys_app"
  instance = google_sql_database_instance.postgres.name
  password = random_password.db_password.result
}