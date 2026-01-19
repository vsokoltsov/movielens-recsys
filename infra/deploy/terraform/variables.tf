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
  default = "csv"
}

variable "rating_threshold" {
  type    = number
  default = 4
}

variable "raw_bucket_name" {
  type    = string
  default = ""
}

variable "models_bucket_name" {
  type    = string
  default = ""
}