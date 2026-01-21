variable "project_id" { type = string }
variable "region"     { type = string }
variable "zone"       { type = string }

variable "image" {
  type        = string
  description = "Full image URI in Artifact Registry"
}

variable "k8s_namespace" {
  type    = string
  default = "recsys"
}

variable "service_name" {
  type    = string
  default = "recsys-api"
}

variable "replicas" {
  type    = number
  default = 1
}

variable "container_port" {
  type    = number
  default = 8000
}

variable "apply_k8s" {
  type        = bool
  description = "Apply Kubernetes manifests via Terraform"
  default     = false
}

variable "model_type" {
  type    = string
  default = "als"
}

variable "model_name" {
  type    = string
  default = "alternating_least_squares.npz"
}

variable "data_source" {
  type    = string
  default = "db"
}

variable "rating_threshold" {
  type    = number
  default = 4
}

variable "db_name" {
  type    = string
  default = "recsys"
}

variable "db_user" {
  type    = string
  default = "recsys_app"
}

variable "db_port" {
  type    = number
  default = 5432
}

variable "apply_migrations" {
  type    = bool
  default = false
}

variable "migrate_run_id" {
  type        = string
  description = "Unique suffix to force a new Job each run (e.g. unix timestamp)"
  default     = ""
}