resource "kubernetes_manifest" "namespace" {
  count    = var.apply_k8s ? 1 : 0
  manifest = local.ns_manifest

  depends_on = [google_container_node_pool.primary]
}

resource "kubernetes_manifest" "deployment" {
  count    = var.apply_k8s ? 1 : 0
  manifest = local.deployment_manifest

  depends_on = [
    kubernetes_manifest.namespace,
    kubernetes_manifest.configmap,
    kubernetes_manifest.ksa,
    kubernetes_manifest.db_secret
  ]

  field_manager {
    name            = "terraform"
    force_conflicts = true
  }

   computed_fields = [
    "spec.template.spec.containers[0].env",
    "spec.template.spec.serviceAccount"
  ]
}

resource "kubernetes_manifest" "service" {
  count    = var.apply_k8s ? 1 : 0
  manifest = local.service_manifest

  depends_on = [kubernetes_manifest.deployment]
}

resource "kubernetes_manifest" "configmap" {
  count    = var.apply_k8s ? 1 : 0
  manifest = local.configmap_manifest

  depends_on = [kubernetes_manifest.namespace]

  field_manager {
    name            = "terraform"
    force_conflicts = true
  }
}

data "kubernetes_service_v1" "svc" {
  count = var.apply_k8s ? 1 : 0

  metadata {
    name      = var.service_name
    namespace = var.k8s_namespace
  }

  depends_on = [kubernetes_manifest.service]
}

locals {
  ksa_name = "recsys-api"
}

resource "kubernetes_manifest" "ksa" {
  count = var.apply_k8s ? 1 : 0

  manifest = {
    apiVersion = "v1"
    kind       = "ServiceAccount"
    metadata = {
      name      = local.ksa_name
      namespace = var.k8s_namespace
      annotations = {
        "iam.gke.io/gcp-service-account" = google_service_account.recsys_cloudsql.email
      }
    }
  }

  depends_on = [kubernetes_manifest.namespace]
}

resource "kubernetes_manifest" "migrate_job" {
  count = var.apply_k8s && var.apply_migrations ? 1 : 0

  manifest = local.migrate_job_manifest

  computed_fields = [
    "spec.template.metadata.labels"
  ]

  depends_on = [
    kubernetes_manifest.namespace,
    kubernetes_manifest.configmap,
    kubernetes_manifest.db_secret,
    kubernetes_manifest.ksa,
  ]
}