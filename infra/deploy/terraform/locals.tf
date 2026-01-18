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
                }
              ]
            }
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
    }
  }
}