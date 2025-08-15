# Additional nodes for non-GPU work 
# Here we're creating a number of general purpose nodes for non-GPU work
# This might be "simple" data engineering or server work
# GPU Worker nodes should have the GPU taint on them so non-GPU jobs won't use them
# This includes Nebius-managed services like mlflow
generic_workers = [{
  min_size                    = 2
  max_size                    = 2
  max_unavailable_percent = 50
  base_name = "k8s_generic_worker"
  # max_surge_percent       = 50
  # drain_timeout           = "10s"
  resource = {
    # Platform type: General CPU architecture
    # 	From vendor presets (static or dynamically generated)
    # 		Nebius is either cpu-d3 (AMD Epyc Genoa) or cpu-e2 (Intel Ice Lake)
    # 		AWS examples would be t2 (x86), t4g (Graviton ARM 64) (very region dependent)
    #		NextGen is either arm64.nano, arm64.rk3588, i7.mini, nv.orin, nv.orin.nano, nv.orin.nx, nv.orin.switch, nv.thor
    platform = "cpu-e2"		# From presets; Nebius is either cpu-d3 (AMD Epyc Genoa) or cpu-e2 (Intel Ice Lake)
    # The size of the instance (all RAM sizes are approximate; round down)
    #   From vendor presets 
    # 		Nebius is vcpu-ram in with 4gb per vcpu, 2vpcu-8gb, 4vcpu-16gb, 8vcpu-32gb, 16vcpu-64gb
    # 		AWS examples would be nano (1vcpu,0.5GB), micro (1 vcpu, 1gb), small (1x2gb), medium (2x4), large (2x8), xlarge (4x16), 2xlarge (8x32)
    #		NextGen is core-ram: 1c-1gb, 1c-2gb, 1c-4gb, 2c-4gb, 2c-8gb, 4c-8gb, 4c-16gb, 4c-32gb, 8c-32gb, 16c-64gb
    #			(note NextGen only counts "big" cores in ARM big.little configuration excet in 1c-1gb)
    preset   = "4vcpu-16gb"	# vcpu-ram in with 4gb per vcpu, 2vpcu-8gb, 4vcpu-16gb, 8vcpu-32gb, 16vcpu-64gb
  }

  # The boot disk type for the operating system
  # Boot disks are vendor-specific
  #    Nebius is NETWORK_SSD, NETWORK_SSD_NON_REPLICATED, NETWORK_SSD_IO_M3: https://docs.nebius.com/compute/storage/types
  #    NextGen is LOCAL, NFS, or ISCSI (with the respective pre-defined types determining performance)
  boot_disk = {
    type                 = "NETWORK_SSD_NON_REPLICATED"
    size_gibibytes       = 1023
    block_size_kibibytes = 64
  }

  # Local Cache Disk
  # These are typically local NVME disks that with ephemeral data
  # Do NOT expect anything stored on these disks to survive - they're only for performance purposes
  # Typical use would be to download a read-only dataset from Delta Lake for ML training or DB Sharding
  cache_disks = [
	  {
	    # The local device in /dev
	    disk_name = "/dev/nvme01p1"
	    # The reserved size. Won't exceed the requested size; smaller caches are less likely to be purged
	    size = "1tb"
	    # Where to expect the final mount point for your apps to use.
	    # The actual storage might be manipulated beforehand (for example, magic behind the scenes might give you an overlay image but it will be on the device, type, and size)
	    mount_point = "/mnt/local_cache"
	    fs_type = "ext4"
	    cache_type = "rsync"
	    # Mount options are typically "ro" (read-only) or "rw" (read-write)
	    mount_options = "rw" 
	  }
  ]

  # The shared filesystems to mount
  #    These are assumed to be pre-created
  #    Typical types might be DeltaLake (DBFS-like), 
  #    Actual type will be specified by the mount itself
  shared_filesystens = [
	{
		shared_filesystem_id = ""
		shared_filesystem_mount = "/mnt/share"
	}
  ]

}]

