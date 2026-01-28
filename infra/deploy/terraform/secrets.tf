resource "kubernetes_manifest" "db_secret" {
  count    = var.apply_k8s ? 1 : 0
  manifest = local.secret_manifest

  depends_on = [kubernetes_manifest.namespace]

  field_manager {
    name            = "terraform"
    force_conflicts = true
  }
  computed_fields = [
    "data",
    "stringData",
  ]
}