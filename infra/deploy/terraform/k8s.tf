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
    kubernetes_manifest.configmap
  ]

  field_manager {
    name            = "terraform"
    force_conflicts = true
  }

   computed_fields = [
    "spec.template.spec.containers[0].env"
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