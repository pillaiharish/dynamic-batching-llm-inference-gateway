output "laboratory_spec" {
  description = "Non-secret specification used to create the experiment laboratory."
  value       = terraform_data.h100_lab.output
}
