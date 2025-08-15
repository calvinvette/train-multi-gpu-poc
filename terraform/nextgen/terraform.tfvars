#----------------------------------------------------------------------------------------------------------------------#
#                                                                                                                      #
#                                                                                                                      #
#                                              Terraform - example values                                              #
#                                                                                                                      #
#                                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------#

# Name of the company. It is used for context name of the cluster in .kubeconfig file.
company_name = "nextgen"

#----------------------------------------------------------------------------------------------------------------------#
#                                                                                                                      #
#                                                                                                                      #
#                                                    Infrastructure                                                    #
#                                                                                                                      #
#                                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------#
# region Infrastructure

#----------------------------------------------------------------------------------------------------------------------#
#                                                                                                                      #
#                                                        Storage                                                       #
#                                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------#
# region Storage

# Shared filesystem to be used on controller nodes.
# ---
filestore_controller_spool = {
  spec = {
    size_gibibytes       = 128
    block_size_kibibytes = 16
  }
}
# Or use existing filestore.
# ---
# filestore_controller_spool = {
#   existing = {
#     id = "computefilesystem-e00z0fsxcmmfss4jh4"
#   }
# }

# Shared filesystem to be used on controller, worker, and login nodes.
# Notice that auto-backups are enabled for filesystems with size less than 12 TiB.
# If you need backups for jail larger than 12 TiB, set 'backups_enabled' to 'force_enable' down below.
# ---
# filestore_jail = {
#   spec = {
#     size_gibibytes       = 2048
#     block_size_kibibytes = 4
#   }
# }
# Or use existing filestore.
# ---
filestore_jail = {
  existing = {
    id = "computefilesystem-e00z0fsxcmmfss4jh4"
    size_bytes = 2199023255552
  }
}

# Additional (Optional) shared filesystems to be mounted inside jail.
# If a big filesystem is needed it's better to deploy this additional storage because jails bigger than 12 TiB
# ARE NOT BACKED UP by default.
# ---
# filestore_jail_submounts = [{
#   name       = "data"
#   mount_path = "/mnt/data"
#   spec = {
#     size_gibibytes       = 2048
#     block_size_kibibytes = 4
#   }
# }]
# Or use existing filestores.
# ---
# filestore_jail_submounts = [{
#   name       = "data"
#   mount_path = "/mnt/data"
#   existing = {
#     id = "computefilesystem-e00z0fsxcmmfss4jh4"
#     size_bytes = 2199023255552
#   }
# }]

# Additional (Optional) node-local Network-SSD disks to be mounted inside jail on worker nodes.
# It will create compute disks with provided spec for each node via CSI.
# NOTE: in case of `NETWORK_SSD_NON_REPLICATED` disk type, `size` must be divisible by 93Gi - https://docs.nebius.com/compute/storage/types#disks-types.
# ---
# node_local_jail_submounts = []
# ---
# node_local_jail_submounts = [{
#   name            = "local-data"
#   mount_path      = "/mnt/local-data"
#   size_gibibytes  = 2048
#   disk_type       = "NETWORK_SSD"
#   filesystem_type = "ext4"
# }]

# Whether to create extra NRD disks for storing Docker/Enroot images and container filesystems on each worker node.
# It will create compute disks with provided spec for each node via CSI.
# NOTE: In case you're not going to use Docker/Enroot in your workloads, it's worth disabling this feature.
# NOTE: `size` must be divisible by 93Gi - https://docs.nebius.com/compute/storage/types#disks-types.
# ---
node_local_image_disk = {
  enabled = false
}
# ---
# node_local_image_disk = {
#   enabled = true
#   spec = {
#     size_gibibytes  = 930
#     filesystem_type = "ext4"
#     # Could be changed to `NETWORK_SSD_NON_REPLICATED`
#     disk_type = "NETWORK_SSD_NON_REPLICATED"
#   }
# }

# Shared filesystem to be used for accounting DB.
# By default, null.
# Required if accounting_enabled is true.
# ---
filestore_accounting = {
  spec = {
    size_gibibytes       = 128
    block_size_kibibytes = 16
  }
}
# Or use existing filestore.
# ---
# filestore_accounting = {
#   existing = {
#     id = "computefilesystem-e00z0fsxcmmfss4jh4"
#   }
# }

# endregion Storage

# region nfs-server

nfs = {
  enabled        = true
  size_gibibytes = 1023
  mount_path     = "/home" 
  name     = "slurm_nfs_server"
  resource = {
    platform = "cpu-e2"
    preset   = "4vcpu-16gb"
  }
  public_ip = false
    ssh_user_name   = "calvin"
    ssh_public_keys = [
      "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC7bRa6eOwSuqFTGFQpmkGauCkiCSIWlBe3lRAu9Q/0SbNurL7bekuFHk/HvCMCpe09DOJ3Cw2wZ6nJstfvwPe4nTxUsd9G4Mk+tnSiy1RG7IMYSZoJj9gpzG8mDir0VRora6bwZNeq1Rz6PZrdkKFYBcGFGp489Skf2+Ia3eV6GEIt7vLKvrb+QzPpXDQrJv4L7XvJep/AoK7yq3duHsDz6vN1Fx2rqr8xssxXHeNT1TpKrsiSimO+Nzb1K5S8NM22MX0FEJYcNlGkt0CRHyO37VNoFE4K+Ulg6YzrtD8Jtqh+FHKXcdL1OawMIPPdjiYgVmqwyFLPiw/eQssQ1nDVCYqb7H/aTwVX4F9zdhj33DAUtOy9EHk9/HwvWZqomv+/JnSXVcDu4y3i3GQtjWiz0bEMK7HcfNu7RxkH1GxSdyhgSpAWIt6gPTX0NuCfbs4+wWFBBBLgeeL9T8Sqn8zJS23X0eZ2o11yW16GqIGmVPYgatksdqfnIz7XoM7/EuU= calvin@orin1_slurm",
      "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCygPCLKxhzSXREY+FeEf10YtjLFRb2ndaRZ2mW7Kn9rFBjJ5+q3dwVxa3nUQVOoOfS+v0eV1qyrsBbwlKBA3J<JFl9T2k1umjQ51TacTNKFdk36TMJI/rTptQddlYk/u6vP2r5qDu80s6The/99N/7tKc/XFVp1yVoYUYaQzzRWbc+TGPl9iF2uqSSbJFRjzu9wU9JnGJkIsrZB0ccwmeGg078lC1auA2DLtPLw+bYFmcuWlPcFwve21O9iOVKDHH0KI595lojbpoK/TNOXcQumpuu11fo9fsUu/twH7zejrcHjWSB+CHLoU4PxuuEsd3pVQt3qhVC7amFnnpdS/0M3mQdBpFe6T61PcaI+s7Ij2CLmfNtGDuntsmx0xodJrgd/nYZxTVTq7MNKRBvI2Huptbe+Lnw8Nzei8RCRj9GKgFmt8oR/Rdahf/AEXITUrCvnfF//VdjgHCtEmoKtGe913cYuKf8aTq4XD/Pxifrr3tP6QRNVmbWafPgaY3Mj888= calvin@iPadPro2023"
    ]
}

# endregion nfs-server

#----------------------------------------------------------------------------------------------------------------------#
#                                                                                                                      #
#                                                                                                                      #
#                                                         Slurm                                                        #
#                                                                                                                      #
#                                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------#
# region Slurm

# Version of soperator.
# ---
slurm_operator_version = "1.21.10"

# Is the version of soperator stable or not.
# ---
slurm_operator_stable = true

# Type of the Slurm partition config. Could be either `default` or `custom`.
# By default, "default".
# ---
slurm_partition_config_type = "default"

# Partition config in case of `custom` slurm_partition_config_type.
# Each string must be started with `PartitionName`.
# By default, empty list.
# ---
# slurm_partition_raw_config = [
#   "PartitionName=low_priority Nodes=low_priority Default=YES MaxTime=INFINITE State=UP PriorityTier=1",
#   "PartitionName=high_priority Nodes=low_priority Default=NO MaxTime=INFINITE State=UP PriorityTier=2"
# ]
# If Nodes present, they must not contain node names: use only nodeset values, "ALL" or "".
# If nodesets are used in the partition config, slurm_worker_features with non-empty nodeset_name
# must be declared (see below).
# Specifying specific nodes is not supported since Dynamic Nodes are used.
# For more details, see https://slurm.schedmd.com/dynamic_nodes.html#partitions.

# List of features to be enabled on worker nodes. Each feature object has:
# - name: (Required) The name of the feature.
# - hostlist_expr: (Required) A Slurm hostlist expression, e.g. "workers-[0-2,10],workers-[3-5]".
#   Soperator will run these workers with the feature name.
# - nodeset_name: (Optional) The Slurm nodeset name to be provisioned using this feature.
#   This nodeset may be used in conjunction with partitions.
#   It is required if `Nodes=<nodeset_name>` is used for a partition.
#
# slurm_worker_features = [
#   {
#     name = "low_priority"
#     hostlist_expr = "worker-[0-0]"
#     nodeset_name = "low_priority"
#   },
#   {
#     name = "low_priority"
#     hostlist_expr = "worker-1"
#     nodeset_name = "high_priority"
#   }
# ]

# Health check config:
# - health_check_interval: (Required) Interval for health check run in seconds.
# - health_check_program: (Required) Program for health check run.
# - health_check_node_state: (Required) What node states should execute the program.
#
# slurm_health_check_config = {
#   health_check_interval: 30,
#   health_check_program: "/usr/bin/gpu_healthcheck.sh",
#   health_check_node_state: [
#     {
#       state: "ANY"
#     },
#     {
#       state: "CYCLE"
#     }
#   ]
# }

#----------------------------------------------------------------------------------------------------------------------#
#                                                                                                                      #
#                                                         Nodes                                                        #
#                                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------#
# region Nodes

# Configuration of System node set for system resources created by Soperator.
# Keep in mind that the k8s nodegroup will have auto-scaling enabled and the actual number of nodes depends on the size
# of the cluster.
# ---
slurm_nodeset_system = {
  min_size = 1
  max_size = 1
  base_name = "slurm_nodeset_system_node"
  resource = {
    platform = "cpu-e2"
    preset   = "4vcpu-16gb"
  }
  boot_disk = {
    type                 = "NETWORK_SSD_NON_REPLICATED"
    size_gibibytes       = 186 # Has to be > 128 and multiple of 93 if NON_REPLICATED
    block_size_kibibytes = 16
  }
}

# Configuration of Slurm Controller node set.
# ---
slurm_nodeset_controller = {
  size = 1
  name = "slurm_nodeset_controller"
  resource = {
    platform = "cpu-e2"
    preset   = "4vcpu-16gb"
  }
  boot_disk = {
    type                 = "NETWORK_SSD_NON_REPLICATED"
    size_gibibytes       = 186 # Has to be > 128 and multiple of 93 if NON_REPLICATED
    block_size_kibibytes = 16
  }
}

# Configuration of Slurm Worker node sets.
# There can be only one Worker node set for a while.
# nodes_per_nodegroup allows you to split node set into equally-sized node groups to keep your cluster accessible and working
# during maintenance. Example: nodes_per_nodegroup=3 for size=12 nodes will create 4 groups with 3 nodes in every group.
# infiniband_fabric is required field
# ---
slurm_nodeset_workers = [{
  size                    = 2
  nodes_per_nodegroup     = 1
  max_unavailable_percent = 50
  base_name = "slurm_nodeset_worker_node"
  # max_surge_percent       = 50
  # drain_timeout           = "10s"
  resource = {
    platform = "gpu-h100-sxm"
    preset   = "1gpu-16vcpu-200gb"
  }
  boot_disk = {
    type                 = "NETWORK_SSD_NON_REPLICATED"
    size_gibibytes       = 1023
    block_size_kibibytes = 64
  }
  # The fabric choices depends on the location (region) and type of machine node they're connected to
  # We're using gpu-h100-sxm in eu-north1, so that gives us a choice of 
  # fabric-2, fabric-3, fabric-4, or fabric-6
  # https://docs.nebius.com/compute/clusters/gpu
  # However, IB is only availble on the 8gpu machines, not the 1gpu
  # gpu_cluster = {
  #   infiniband_fabric = "fabric-6"
  # }
}]

# Configuration of Slurm Login node set.
# ---
slurm_nodeset_login = {
  size = 1
  labels = {
    name = "slurm_nodeset_login_node"
  } 
  resource = {
    platform = "cpu-e2"
    preset   = "2vcpu-8gb"
  }
  boot_disk = {
    type                 = "NETWORK_SSD_NON_REPLICATED"
    size_gibibytes       = 186 # Has to be > 128 and multiple of 93 if NON_REPLICATED
    block_size_kibibytes = 16
  }
}

# Configuration of Slurm Accounting node set.
# Required in case of Accounting usage.
# By default, null.
# ---
slurm_nodeset_accounting = {
  base_name = "slurm_nodeset_accounting_node"
  resource = {
    platform = "cpu-e2"
    preset   = "2vcpu-8gb"
  }
  boot_disk = {
    type                 = "NETWORK_SSD"
    size_gibibytes       = 128
    block_size_kibibytes = 16
  }
}

#----------------------------------------------------------------------------------------------------------------------#
#                                                         Login                                                        #
#----------------------------------------------------------------------------------------------------------------------#
# region Login

# Authorized keys accepted for connecting to Slurm login nodes via SSH as 'root' user.
# ---
slurm_login_ssh_root_public_keys = [
  "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC7bRa6eOwSuqFTGFQpmkGauCkiCSIWlBe3lRAu9Q/0SbNurL7bekuFHk/HvCMCpe09DOJ3Cw2wZ6nJstfvwPe4nTxUsd9G4Mk+tnSiy1RG7IMYSZoJj9gpzG8mDir0VRora6bwZNeq1Rz6PZrdkKFYBcGFGp489Skf2+Ia3eV6GEIt7vLKvrb+QzPpXDQrJv4L7XvJep/AoK7yq3duHsDz6vN1Fx2rqr8xssxXHeNT1TpKrsiSimO+Nzb1K5S8NM22MX0FEJYcNlGkt0CRHyO37VNoFE4K+Ulg6YzrtD8Jtqh+FHKXcdL1OawMIPPdjiYgVmqwyFLPiw/eQssQ1nDVCYqb7H/aTwVX4F9zdhj33DAUtOy9EHk9/HwvWZqomv+/JnSXVcDu4y3i3GQtjWiz0bEMK7HcfNu7RxkH1GxSdyhgSpAWIt6gPTX0NuCfbs4+wWFBBBLgeeL9T8Sqn8zJS23X0eZ2o11yW16GqIGmVPYgatksdqfnIz7XoM7/EuU= calvin@orin1_slurm",
  "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCygPCLKxhzSXREY+FeEf10YtjLFRb2ndaRZ2mW7Kn9rFBjJ5+q3dwVxa3nUQVOoOfS+v0eV1qyrsBbwlKBA3J<JFl9T2k1umjQ51TacTNKFdk36TMJI/rTptQddlYk/u6vP2r5qDu80s6The/99N/7tKc/XFVp1yVoYUYaQzzRWbc+TGPl9iF2uqSSbJFRjzu9wU9JnGJkIsrZB0ccwmeGg078lC1auA2DLtPLw+bYFmcuWlPcFwve21O9iOVKDHH0KI595lojbpoK/TNOXcQumpuu11fo9fsUu/twH7zejrcHjWSB+CHLoU4PxuuEsd3pVQt3qhVC7amFnnpdS/0M3mQdBpFe6T61PcaI+s7Ij2CLmfNtGDuntsmx0xodJrgd/nYZxTVTq7MNKRBvI2Huptbe+Lnw8Nzei8RCRj9GKgFmt8oR/Rdahf/AEXITUrCvnfF//VdjgHCtEmoKtGe913cYuKf8aTq4XD/Pxifrr3tP6QRNVmbWafPgaY3Mj888= calvin@iPadPro2023"
]

# endregion Login

#----------------------------------------------------------------------------------------------------------------------#
#                                                       Exporter                                                       #
#----------------------------------------------------------------------------------------------------------------------#
# region Exporter

# Whether to enable Slurm metrics exporter.
# By default, true.
# ---
slurm_exporter_enabled = true

# endregion Exporter

# endregion Nodes

#----------------------------------------------------------------------------------------------------------------------#
#                                                                                                                      #
#                                                        Config                                                        #
#                                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------#
# region Config

# Shared memory size for Slurm controller and worker nodes in GiB.
# By default, 64.
# ---
slurm_shared_memory_size_gibibytes = 64

# endregion Config
#----------------------------------------------------------------------------------------------------------------------#
#                                                                                                                      #
#                                                       Telemetry                                                      #
#                                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------#
# region Telemetry

# Whether to enable telemetry.
# By default, true.
# ---
telemetry_enabled = true

# Whether to enable dcgm job mapping (adds hpc_job label on DCGM_ metrics).
# By default, true.
# ---
dcgm_job_mapping_enabled = true

public_o11y_enabled = true

# endregion Telemetry

#----------------------------------------------------------------------------------------------------------------------#
#                                                                                                                      #
#                                                       Accounting                                                     #
#                                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------#
# region Accounting

# Whether to enable Accounting.
# By default, true.
# ---
accounting_enabled = false

# endregion Accounting

# endregion Slurm

#----------------------------------------------------------------------------------------------------------------------#
#                                                                                                                      #
#                                                       Backups                                                        #
#                                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------#
# region Backups

# Whether to enable Backups. Choose from 'auto', 'force_enable', 'force_disable'.
# 'auto' turns backups on for jails with max size less than 12 TB and is a default option.
# ---
backups_enabled = "force_disable"

# Password to be used for encrypting jail backups.
# ---
backups_password = "password"

# Cron schedule for backup task.
# See https://docs.k8up.io/k8up/references/schedule-specification.html for more info.
# ---
backups_schedule = "@daily-random"

# Cron schedule for prune task (when old backups are discarded).
# See https://docs.k8up.io/k8up/references/schedule-specification.html for more info.
# ---
backups_prune_schedule = "@daily-random"

# Backups retention policy - how many last automatic backups to save.
# Helps to save storage and to get rid of old backups as they age.
# Manually created backups (without autobackup tag) are not discarded.
#
# You can set keepLast, keepHourly, keepDaily, keepWeekly, keepMonthly and keepYearly.
# ---
backups_retention = {
  # How many daily snapshots to save.
  # ---
  keepDaily = 7
}

# Whether to delete on destroy all backup data from bucket or not.
cleanup_bucket_on_destroy = false

# endregion Backups

#----------------------------------------------------------------------------------------------------------------------#
#                                                                                                                      #
#                                                      Kubernetes                                                      #
#                                                                                                                      #
#----------------------------------------------------------------------------------------------------------------------#
# region k8s

# Version of the k8s to be used.
# Set to null or don't set to use Nebius default (recommended), or specify explicitly
# "1.33": k8s version is not supported, available versions: 1.30.7, 1.31.9
# ---
k8s_version = "1.31"

# SSH user credentials for accessing k8s nodes.
# That option add public ip address to every node.
# By default, empty list.
# ---
# k8s_cluster_node_ssh_access_users = [{
#   name = "<USER1>"
#   public_keys = [
#     "<ENCRYPTION-METHOD1 HASH1 USER1>",
#     "<ENCRYPTION-METHOD2 HASH2 USER1>",
#   ]
# }]

etcd_cluster_size = 3

# endregion k8s


