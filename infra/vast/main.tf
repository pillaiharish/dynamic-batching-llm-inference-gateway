terraform {
  required_version = ">= 1.4"
}

locals {
  lab_identity = {
    disk_gb   = var.disk_gb
    image     = var.image
    mount     = var.volume_mount_path
    offer_id  = var.offer_id
    onstart   = var.onstart_command
    volume_id = var.volume_id
  }
}

resource "terraform_data" "h100_lab" {
  input            = local.lab_identity
  triggers_replace = local.lab_identity

  lifecycle {
    precondition {
      condition     = var.confirm_billable_resource
      error_message = "Set confirm_billable_resource=true only when ready to rent a billable instance."
    }
  }

  provisioner "local-exec" {
    command = "python3 ${path.module}/vast_lab.py create ${path.module}/.state/instance-id"
    environment = {
      VAST_DISK_GB    = tostring(self.input.disk_gb)
      VAST_IMAGE      = self.input.image
      VAST_MOUNT_PATH = self.input.mount
      VAST_OFFER_ID   = tostring(self.input.offer_id)
      VAST_ONSTART    = self.input.onstart
      VAST_VOLUME_ID  = self.input.volume_id == null ? "" : tostring(self.input.volume_id)
    }
  }

  provisioner "local-exec" {
    when    = destroy
    command = "python3 ${path.module}/vast_lab.py destroy ${path.module}/.state/instance-id"
  }
}
