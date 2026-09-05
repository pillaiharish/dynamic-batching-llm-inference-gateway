# Vast H100 laboratory blueprint

This Terraform root creates and destroys the billable laboratory only. It does not download models,
change T0/T1/T2/T3, restart vLLM between measurements, or run benchmarks.

The official `vastai` CLI must be installed and authenticated in its normal user configuration.
No API key is accepted by Terraform. Select a current offer that independently satisfies one H100
80GB, disk, reliability, and networking requirements, copy `terraform.tfvars.example`, pin the image,
then run `terraform init`, `terraform plan`, and `terraform apply`. Applying requires the explicit
`confirm_billable_resource=true` guard. An optional existing local Vast volume can be linked; it must
be on the selected machine.

The small CLI adapter exists because there is no maintained official Vast Terraform Registry provider.
It records the returned contract ID under ignored `.state/` so `terraform destroy` can release it. Back
up Terraform state and that directory together. If creation succeeds but local state is lost, use
`vastai show instances --raw` and destroy the contract manually.

Model acquisition, verification, and restoration remain Chakra Vault responsibilities. Deployment
scripts establish the runtime topology, and the benchmark campaign performs measurements.
