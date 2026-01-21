locals {
  ns_manifest = {
    apiVersion = "v1"
    kind       = "Namespace"
    metadata = {
      name = var.k8s_namespace
    }
  }

  deployment_manifest = {
    apiVersion = "apps/v1"
    kind       = "Deployment"
    metadata = {
      name      = var.service_name
      namespace = var.k8s_namespace
      labels = {
        app = var.service_name
      }
    }
    spec = {
      replicas = var.replicas
      selector = {
        matchLabels = {
          app = var.service_name
        }
      }
      template = {
        metadata = {
          labels = {
            app = var.service_name
          }
        }
        spec = {
          serviceAccountName = local.ksa_name
          containers = [
            {
              name  = var.service_name
              image = var.image
              ports = [{ containerPort = var.container_port }]
              envFrom = [
                {
                  configMapRef = {
                    name = "recsys-config"
                  }
                },
                {
                  secretRef = {
                    name = "recsys-db-secret"
                  }
                }
              ]
            },
            {
              name  = "cloud-sql-proxy"
              image = "gcr.io/cloud-sql-connectors/cloud-sql-proxy:2.11.0"
              args  = [
                "--address=0.0.0.0",
                "--port=5432",
                "$(CLOUDSQL_CONNECTION_NAME)",
              ]
              env = [
                {
                  name = "CLOUDSQL_CONNECTION_NAME"
                  valueFrom = {
                    secretKeyRef = {
                      name = "recsys-db-secret"
                      key  = "CLOUDSQL_CONNECTION_NAME"
                    }
                  }
                }
              ]
              securityContext = {
                runAsNonRoot = true
              }
            },
          ]
        }
      }
    }
  }

  service_manifest = {
    apiVersion = "v1"
    kind       = "Service"
    metadata = {
      name      = var.service_name
      namespace = var.k8s_namespace
      labels = {
        app = var.service_name
      }
    }
    spec = {
      type = "LoadBalancer"
      selector = {
        app = var.service_name
      }
      ports = [
        {
          port       = 80
          targetPort = var.container_port
          protocol   = "TCP"
        }
      ]
    }
  }

  configmap_manifest = {
    apiVersion = "v1"
    kind       = "ConfigMap"
    metadata = {
      name      = "recsys-config"
      namespace = var.k8s_namespace
    }
    data = {
      MOVIELENS_PATH    = "/app/data/raw/ml-1m"
      MODELS            = "/app/data/models"
      MODEL_TYPE        = var.model_type
      MODEL_NAME        = var.model_name
      SOURCE            = var.data_source
      RATING_THRESHOLD  = tostring(var.rating_threshold)
      PROJECT_ID        = var.project_id
      MODEL_BUCKET      = google_storage_bucket.models.name
      RAW_BUCKET        = google_storage_bucket.raw.name 
      DB_HOST = "127.0.0.1"
      DB_PORT = "5432"
      DB_NAME = var.db_name
      DB_USER = var.db_user
    }
  }

  secret_manifest = {
    apiVersion = "v1"
    kind       = "Secret"
    metadata = {
      name      = "recsys-db-secret"
      namespace = var.k8s_namespace
    }
    type = "Opaque"
    data = {
      DB_USER                 = base64encode(var.db_user)
      DB_PASSWORD             = base64encode(random_password.db_password.result)
      DB_NAME                 = base64encode(var.db_name)
      CLOUDSQL_CONNECTION_NAME = base64encode(google_sql_database_instance.postgres.connection_name)
    }
  }

  migrate_job_name = "recsys-migrate-${var.migrate_run_id}"

  migrate_job_manifest = {
    apiVersion = "batch/v1"
    kind       = "Job"
    metadata = {
      name      = local.migrate_job_name
      namespace = var.k8s_namespace
      labels = { app = "recsys-migrate" }
    }
    spec = {
      backoffLimit            = 3
      ttlSecondsAfterFinished = 3600

      template = {
        metadata = { labels = { app = "recsys-migrate" } }
        spec = {
          restartPolicy      = "Never"
          serviceAccountName = local.ksa_name

          containers = [
            {
              name            = "migrate"
              image           = var.image
              imagePullPolicy = "Always"

              command = ["/opt/venv/bin/python"]
              args = ["-m", "alembic", "upgrade", "head"]

              envFrom = [
                { configMapRef = { name = "recsys-config" } },
                { secretRef    = { name = "recsys-db-secret" } },
              ]

              env = [
                { name = "DB_HOST", value = "127.0.0.1" },
                { name = "DB_PORT", value = "5432" },
                {
                  name  = "DATABASE_URL"
                  value = "postgresql+psycopg://$(DB_USER):$(DB_PASSWORD)@$(DB_HOST):$(DB_PORT)/$(DB_NAME)"
                },
              ]
            },
            {
              name  = "cloud-sql-proxy"
              image = "gcr.io/cloud-sql-connectors/cloud-sql-proxy:2.11.0"
              args  = [
                "--address=0.0.0.0",
                "--port=5432",
                "$(CLOUDSQL_CONNECTION_NAME)",
              ]
              env = [
                {
                  name = "CLOUDSQL_CONNECTION_NAME"
                  valueFrom = {
                    secretKeyRef = {
                      name = "recsys-db-secret"
                      key  = "CLOUDSQL_CONNECTION_NAME"
                    }
                  }
                }
              ]
              securityContext = {
                runAsNonRoot = true
              }
            },
          ]
        }
      }
    }
  }
}