<!-- BEGIN_TF_DOCS -->
## Requirements

| Name | Version |
|------|---------|
| <a name="requirement_terraform"></a> [terraform](#requirement\_terraform) | >=1.8.0 |
| <a name="requirement_flux"></a> [flux](#requirement\_flux) | >= 1.5 |
| <a name="requirement_helm"></a> [helm](#requirement\_helm) | <3.0.0 |
| <a name="requirement_nebius"></a> [nebius](#requirement\_nebius) | >=0.4 |
| <a name="requirement_units"></a> [units](#requirement\_units) | >=1.1.1 |

## Providers

| Name | Version |
|------|---------|
| <a name="provider_nebius"></a> [nebius](#provider\_nebius) | 0.5.94 |
| <a name="provider_terraform"></a> [terraform](#provider\_terraform) | n/a |

## Modules

| Name | Source | Version |
|------|--------|---------|
| <a name="module_active_checks"></a> [active\_checks](#module\_active\_checks) | ../../modules/active_checks | n/a |
| <a name="module_backups"></a> [backups](#module\_backups) | ../../modules/backups | n/a |
| <a name="module_backups_store"></a> [backups\_store](#module\_backups\_store) | ../../modules/backups_store | n/a |
| <a name="module_cleanup"></a> [cleanup](#module\_cleanup) | ../../modules/cleanup | n/a |
| <a name="module_filestore"></a> [filestore](#module\_filestore) | ../../modules/filestore | n/a |
| <a name="module_fluxcd"></a> [fluxcd](#module\_fluxcd) | ../../modules/fluxcd | n/a |
| <a name="module_k8s"></a> [k8s](#module\_k8s) | ../../modules/k8s | n/a |
| <a name="module_k8s_storage_class"></a> [k8s\_storage\_class](#module\_k8s\_storage\_class) | ../../modules/k8s/storage_class | n/a |
| <a name="module_login_script"></a> [login\_script](#module\_login\_script) | ../../modules/login | n/a |
| <a name="module_nfs-server"></a> [nfs-server](#module\_nfs-server) | ../../../modules/nfs-server | n/a |
| <a name="module_nvidia_operator_gpu"></a> [nvidia\_operator\_gpu](#module\_nvidia\_operator\_gpu) | ../../../modules/gpu-operator | n/a |
| <a name="module_nvidia_operator_network"></a> [nvidia\_operator\_network](#module\_nvidia\_operator\_network) | ../../../modules/network-operator | n/a |
| <a name="module_o11y"></a> [o11y](#module\_o11y) | ../../modules/o11y | n/a |
| <a name="module_resources"></a> [resources](#module\_resources) | ../../modules/available_resources | n/a |
| <a name="module_slurm"></a> [slurm](#module\_slurm) | ../../modules/slurm | n/a |

## Resources

| Name | Type |
|------|------|
| [terraform_data.check_nfs](https://registry.terraform.io/providers/hashicorp/terraform/latest/docs/resources/data) | resource |
| [terraform_data.check_region](https://registry.terraform.io/providers/hashicorp/terraform/latest/docs/resources/data) | resource |
| [terraform_data.check_slurm_nodeset](https://registry.terraform.io/providers/hashicorp/terraform/latest/docs/resources/data) | resource |
| [terraform_data.check_slurm_nodeset_accounting](https://registry.terraform.io/providers/hashicorp/terraform/latest/docs/resources/data) | resource |
| [terraform_data.check_variables](https://registry.terraform.io/providers/hashicorp/terraform/latest/docs/resources/data) | resource |
| nebius_compute_v1_filesystem.existing_jail | data source |
| nebius_iam_v1_project.this | data source |
| nebius_iam_v1_tenant.this | data source |
| nebius_vpc_v1_subnet.this | data source |

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| <a name="input_accounting_enabled"></a> [accounting\_enabled](#input\_accounting\_enabled) | Whether to enable accounting. | `bool` | `false` | no |
| <a name="input_backups_enabled"></a> [backups\_enabled](#input\_backups\_enabled) | Whether to enable jail backups. Choose from 'auto', 'force\_enable' and 'force\_disable'. 'auto' enables backups for jails with max size < 12 TB. | `string` | `"auto"` | no |
| <a name="input_backups_password"></a> [backups\_password](#input\_backups\_password) | Password for encrypting jail backups. | `string` | n/a | yes |
| <a name="input_backups_prune_schedule"></a> [backups\_prune\_schedule](#input\_backups\_prune\_schedule) | Cron schedule for prune task. | `string` | n/a | yes |
| <a name="input_backups_retention"></a> [backups\_retention](#input\_backups\_retention) | Backups retention policy. | `map(any)` | n/a | yes |
| <a name="input_backups_schedule"></a> [backups\_schedule](#input\_backups\_schedule) | Cron schedule for backup task. | `string` | n/a | yes |
| <a name="input_cleanup_bucket_on_destroy"></a> [cleanup\_bucket\_on\_destroy](#input\_cleanup\_bucket\_on\_destroy) | Whether to delete on destroy all backup data from bucket or not | `bool` | n/a | yes |
| <a name="input_company_name"></a> [company\_name](#input\_company\_name) | Name of the company. It is used for naming Slurm & K8s clusters. | `string` | n/a | yes |
| <a name="input_dcgm_job_mapping_enabled"></a> [dcgm\_job\_mapping\_enabled](#input\_dcgm\_job\_mapping\_enabled) | Whether to enable HPC job mapping by installing a separate dcgm-exporter | `bool` | `true` | no |
| <a name="input_etcd_cluster_size"></a> [etcd\_cluster\_size](#input\_etcd\_cluster\_size) | Size of the etcd cluster. | `number` | `3` | no |
| <a name="input_filestore_accounting"></a> [filestore\_accounting](#input\_filestore\_accounting) | Shared filesystem to be used for accounting DB | <pre>object({<br/>    existing = optional(object({<br/>      id = string<br/>      size_bytes = number<br/>    }))<br/>    spec = optional(object({<br/>      size_gibibytes       = number<br/>      block_size_kibibytes = number<br/>    }))<br/>  })</pre> | `null` | no |
| <a name="input_filestore_controller_spool"></a> [filestore\_controller\_spool](#input\_filestore\_controller\_spool) | Shared filesystem to be used on controller nodes. | <pre>object({<br/>    existing = optional(object({<br/>      id = string<br/>      size_bytes = number<br/>    }))<br/>    spec = optional(object({<br/>      size_gibibytes       = number<br/>      block_size_kibibytes = number<br/>    }))<br/>  })</pre> | n/a | yes |
| <a name="input_filestore_jail"></a> [filestore\_jail](#input\_filestore\_jail) | Shared filesystem to be used on controller, worker, and login nodes. | <pre>object({<br/>    existing = optional(object({<br/>      id = string<br/>      size_bytes = number<br/>    }))<br/>    spec = optional(object({<br/>      size_gibibytes       = number<br/>      block_size_kibibytes = number<br/>    }))<br/>  })</pre> | n/a | yes |
| <a name="input_filestore_jail_submounts"></a> [filestore\_jail\_submounts](#input\_filestore\_jail\_submounts) | Shared filesystems to be mounted inside jail. | <pre>list(object({<br/>    name       = string<br/>    mount_path = string<br/>    existing = optional(object({<br/>      id = string<br/>      size_bytes = number<br/>    }))<br/>    spec = optional(object({<br/>      size_gibibytes       = number<br/>      block_size_kibibytes = number<br/>    }))<br/>  }))</pre> | `[]` | no |
| <a name="input_flux_interval"></a> [flux\_interval](#input\_flux\_interval) | The interval for Flux to check for changes. | `string` | `"1m"` | no |
| <a name="input_github_org"></a> [github\_org](#input\_github\_org) | The GitHub organization. | `string` | `"nebius"` | no |
| <a name="input_github_repository"></a> [github\_repository](#input\_github\_repository) | The GitHub repository. | `string` | `"soperator"` | no |
| <a name="input_iam_project_id"></a> [iam\_project\_id](#input\_iam\_project\_id) | ID of the IAM project. | `string` | n/a | yes |
| <a name="input_iam_tenant_id"></a> [iam\_tenant\_id](#input\_iam\_tenant\_id) | ID of the IAM tenant. | `string` | n/a | yes |
| <a name="input_iam_token"></a> [iam\_token](#input\_iam\_token) | IAM token used for communicating with Nebius services. | `string` | n/a | yes |
| <a name="input_k8s_cluster_node_ssh_access_users"></a> [k8s\_cluster\_node\_ssh\_access\_users](#input\_k8s\_cluster\_node\_ssh\_access\_users) | SSH user credentials for accessing k8s nodes. | <pre>list(object({<br/>    name        = string<br/>    public_keys = list(string)<br/>  }))</pre> | `[]` | no |
| <a name="input_k8s_version"></a> [k8s\_version](#input\_k8s\_version) | Version of the k8s to be used. | `string` | `null` | no |
| <a name="input_maintenance"></a> [maintenance](#input\_maintenance) | Whether to enable maintenance mode. | `string` | `"none"` | no |
| <a name="input_nfs"></a> [nfs](#input\_nfs) | n/a | <pre>object({<br/>    enabled        = bool<br/>    size_gibibytes = number<br/>    mount_path     = optional(string, "/home")<br/>    resource = object({<br/>      platform = string<br/>      preset   = string<br/>    })<br/>    public_ip = bool<br/>  })</pre> | <pre>{<br/>  "enabled": false,<br/>  "public_ip": false,<br/>  "resource": {<br/>    "platform": "cpu-d3",<br/>    "preset": "32vcpu-128gb"<br/>  },<br/>  "size_gibibytes": 93<br/>}</pre> | no |
| <a name="input_node_local_image_disk"></a> [node\_local\_image\_disk](#input\_node\_local\_image\_disk) | Whether to create extra NRD/IO M3 disks for storing Docker/Enroot images and container filesystems on each worker node. | <pre>object({<br/>    enabled = bool<br/>    spec = optional(object({<br/>      size_gibibytes  = number<br/>      filesystem_type = string<br/>      disk_type       = string<br/>    }))<br/>  })</pre> | <pre>{<br/>  "enabled": false<br/>}</pre> | no |
| <a name="input_node_local_jail_submounts"></a> [node\_local\_jail\_submounts](#input\_node\_local\_jail\_submounts) | Node-local disks to be mounted inside jail on worker nodes. | <pre>list(object({<br/>    name            = string<br/>    mount_path      = string<br/>    size_gibibytes  = number<br/>    disk_type       = string<br/>    filesystem_type = string<br/>  }))</pre> | `[]` | no |
| <a name="input_o11y_iam_tenant_id"></a> [o11y\_iam\_tenant\_id](#input\_o11y\_iam\_tenant\_id) | ID of the IAM tenant for O11y. | `string` | n/a | yes |
| <a name="input_o11y_profile"></a> [o11y\_profile](#input\_o11y\_profile) | Profile for nebius CLI for public o11y. | `string` | n/a | yes |
| <a name="input_public_o11y_enabled"></a> [public\_o11y\_enabled](#input\_public\_o11y\_enabled) | Whether to enable public observability endpoints. | `bool` | `true` | no |
| <a name="input_region"></a> [region](#input\_region) | Region of the project. | `string` | n/a | yes |
| <a name="input_slurm_accounting_config"></a> [slurm\_accounting\_config](#input\_slurm\_accounting\_config) | Slurm.conf accounting configuration. See https://slurm.schedmd.com/slurm.conf.html. Not all options are supported. | `map(any)` | `{}` | no |
| <a name="input_slurm_exporter_enabled"></a> [slurm\_exporter\_enabled](#input\_slurm\_exporter\_enabled) | Whether to enable Slurm metrics exporter. | `bool` | `true` | no |
| <a name="input_slurm_health_check_config"></a> [slurm\_health\_check\_config](#input\_slurm\_health\_check\_config) | Health check configuration. | <pre>object({<br/>    health_check_interval = number<br/>    health_check_program  = string<br/>    health_check_node_state = list(object({<br/>      state = string<br/>    }))<br/>  })</pre> | `null` | no |
| <a name="input_slurm_login_ssh_root_public_keys"></a> [slurm\_login\_ssh\_root\_public\_keys](#input\_slurm\_login\_ssh\_root\_public\_keys) | Authorized keys accepted for connecting to Slurm login nodes via SSH as 'root' user. | `list(string)` | n/a | yes |
| <a name="input_slurm_login_sshd_config_map_ref_name"></a> [slurm\_login\_sshd\_config\_map\_ref\_name](#input\_slurm\_login\_sshd\_config\_map\_ref\_name) | Name of configmap with SSHD config, which runs in slurmd container. | `string` | `""` | no |
| <a name="input_slurm_nodeset_accounting"></a> [slurm\_nodeset\_accounting](#input\_slurm\_nodeset\_accounting) | Configuration of Slurm Accounting node set. | <pre>object({<br/>    resource = object({<br/>      platform = string<br/>      preset   = string<br/>    })<br/>    boot_disk = object({<br/>      type                 = string<br/>      size_gibibytes       = number<br/>      block_size_kibibytes = number<br/>    })<br/>  })</pre> | `null` | no |
| <a name="input_slurm_nodeset_controller"></a> [slurm\_nodeset\_controller](#input\_slurm\_nodeset\_controller) | Configuration of Slurm Controller node set. | <pre>object({<br/>    size = number<br/>    resource = object({<br/>      platform = string<br/>      preset   = string<br/>    })<br/>    boot_disk = object({<br/>      type                 = string<br/>      size_gibibytes       = number<br/>      block_size_kibibytes = number<br/>    })<br/>  })</pre> | <pre>{<br/>  "boot_disk": {<br/>    "block_size_kibibytes": 4,<br/>    "size_gibibytes": 128,<br/>    "type": "NETWORK_SSD"<br/>  },<br/>  "resource": {<br/>    "platform": "cpu-d3",<br/>    "preset": "16vcpu-64gb"<br/>  },<br/>  "size": 1<br/>}</pre> | no |
| <a name="input_slurm_nodeset_login"></a> [slurm\_nodeset\_login](#input\_slurm\_nodeset\_login) | Configuration of Slurm Login node set. | <pre>object({<br/>    size = number<br/>    resource = object({<br/>      platform = string<br/>      preset   = string<br/>    })<br/>    boot_disk = object({<br/>      type                 = string<br/>      size_gibibytes       = number<br/>      block_size_kibibytes = number<br/>    })<br/>  })</pre> | <pre>{<br/>  "boot_disk": {<br/>    "block_size_kibibytes": 4,<br/>    "size_gibibytes": 128,<br/>    "type": "NETWORK_SSD"<br/>  },<br/>  "resource": {<br/>    "platform": "cpu-d3",<br/>    "preset": "16vcpu-64gb"<br/>  },<br/>  "size": 1<br/>}</pre> | no |
| <a name="input_slurm_nodeset_system"></a> [slurm\_nodeset\_system](#input\_slurm\_nodeset\_system) | Configuration of System node set for system resources created by Soperator. | <pre>object({<br/>    min_size = number<br/>    max_size = number<br/>    resource = object({<br/>      platform = string<br/>      preset   = string<br/>    })<br/>    boot_disk = object({<br/>      type                 = string<br/>      size_gibibytes       = number<br/>      block_size_kibibytes = number<br/>    })<br/>  })</pre> | <pre>{<br/>  "boot_disk": {<br/>    "block_size_kibibytes": 4,<br/>    "size_gibibytes": 128,<br/>    "type": "NETWORK_SSD"<br/>  },<br/>  "max_size": 9,<br/>  "min_size": 3,<br/>  "resource": {<br/>    "platform": "cpu-d3",<br/>    "preset": "16vcpu-64gb"<br/>  }<br/>}</pre> | no |
| <a name="input_slurm_nodeset_workers"></a> [slurm\_nodeset\_workers](#input\_slurm\_nodeset\_workers) | Configuration of Slurm Worker node sets. | <pre>list(object({<br/>    size                    = number<br/>    nodes_per_nodegroup     = number<br/>    max_unavailable_percent = optional(number)<br/>    max_surge_percent       = optional(number)<br/>    drain_timeout           = optional(string)<br/>    resource = object({<br/>      platform = string<br/>      preset   = string<br/>    })<br/>    boot_disk = object({<br/>      type                 = string<br/>      size_gibibytes       = number<br/>      block_size_kibibytes = number<br/>    })<br/>    gpu_cluster = optional(object({<br/>      infiniband_fabric = string<br/>    }))<br/>  }))</pre> | <pre>[<br/>  {<br/>    "boot_disk": {<br/>      "block_size_kibibytes": 4,<br/>      "size_gibibytes": 128,<br/>      "type": "NETWORK_SSD"<br/>    },<br/>    "max_unavailable_percent": 50,<br/>    "nodes_per_nodegroup": 1,<br/>    "resource": {<br/>      "platform": "cpu-d3",<br/>      "preset": "16vcpu-64gb"<br/>    },<br/>    "size": 1<br/>  }<br/>]</pre> | no |
| <a name="input_slurm_operator_stable"></a> [slurm\_operator\_stable](#input\_slurm\_operator\_stable) | Is the version of soperator stable. | `bool` | `true` | no |
| <a name="input_slurm_operator_version"></a> [slurm\_operator\_version](#input\_slurm\_operator\_version) | Version of soperator. | `string` | n/a | yes |
| <a name="input_slurm_partition_config_type"></a> [slurm\_partition\_config\_type](#input\_slurm\_partition\_config\_type) | Type of the Slurm partition config. Could be either `default` or `custom`. | `string` | `"default"` | no |
| <a name="input_slurm_partition_raw_config"></a> [slurm\_partition\_raw\_config](#input\_slurm\_partition\_raw\_config) | Partition config in case of `custom` slurm\_partition\_config\_type. Each string must be started with `PartitionName`. | `list(string)` | `[]` | no |
| <a name="input_slurm_rest_enabled"></a> [slurm\_rest\_enabled](#input\_slurm\_rest\_enabled) | Whether to enable Slurm REST API. | `bool` | `true` | no |
| <a name="input_slurm_shared_memory_size_gibibytes"></a> [slurm\_shared\_memory\_size\_gibibytes](#input\_slurm\_shared\_memory\_size\_gibibytes) | Shared memory size for Slurm controller and worker nodes in GiB. | `number` | `64` | no |
| <a name="input_slurm_worker_features"></a> [slurm\_worker\_features](#input\_slurm\_worker\_features) | List of features to be enabled on worker nodes. | <pre>list(object({<br/>    name          = string<br/>    hostlist_expr = string<br/>    nodeset_name  = optional(string)<br/>  }))</pre> | `[]` | no |
| <a name="input_slurm_worker_sshd_config_map_ref_name"></a> [slurm\_worker\_sshd\_config\_map\_ref\_name](#input\_slurm\_worker\_sshd\_config\_map\_ref\_name) | Name of configmap with SSHD config, which runs in slurmd container. | `string` | `""` | no |
| <a name="input_slurmdbd_config"></a> [slurmdbd\_config](#input\_slurmdbd\_config) | Slurmdbd.conf configuration. See https://slurm.schedmd.com/slurmdbd.conf.html.Not all options are supported. | `map(any)` | `{}` | no |
| <a name="input_telemetry_enabled"></a> [telemetry\_enabled](#input\_telemetry\_enabled) | Whether to enable telemetry. | `bool` | `true` | no |
| <a name="input_use_default_apparmor_profile"></a> [use\_default\_apparmor\_profile](#input\_use\_default\_apparmor\_profile) | Whether to use default AppArmor profile. | `bool` | `true` | no |
| <a name="input_vpc_subnet_id"></a> [vpc\_subnet\_id](#input\_vpc\_subnet\_id) | ID of VPC subnet. | `string` | n/a | yes |

## Outputs

No outputs.
<!-- END_TF_DOCS -->