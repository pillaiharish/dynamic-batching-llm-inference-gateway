variable "offer_id" {
  description = "Current Vast offer ID selected for exactly one H100 80GB instance."
  type        = number
  validation {
    condition     = var.offer_id > 0
    error_message = "offer_id must be positive."
  }
}

variable "image" {
  description = "Pinned container image reference for the experiment laboratory."
  type        = string
}

variable "disk_gb" {
  description = "Ephemeral instance disk size."
  type        = number
  default     = 100
  validation {
    condition     = var.disk_gb >= 50
    error_message = "disk_gb must be at least 50."
  }
}

variable "volume_id" {
  description = "Optional existing Vast volume ID. It must reside on the selected machine."
  type        = number
  default     = null
}

variable "volume_mount_path" {
  description = "Mount point used when volume_id is set."
  type        = string
  default     = "/workspace"
}

variable "onstart_command" {
  description = "Bootstrap metadata only; topology transitions remain deploy-lifecycle work."
  type        = string
  default     = ""
}

variable "confirm_billable_resource" {
  description = "Explicit guard against accidental paid instance creation."
  type        = bool
  default     = false
}
