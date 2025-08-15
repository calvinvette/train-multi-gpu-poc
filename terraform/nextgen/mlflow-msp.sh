#!/bin/sh

# Defaults to first VPC Network ID in the list of VPC networks for the current logged-in user (assuming nebius iam login)
export NEBIUS_VPC_NETWORK_ID=${NEBIUS_VPC_NETWORK_ID:-$(nebius vpc network list | yq -r ".items[0].metadata.id")}
# echo $NEBIUS_VPC_NETWORK_ID

# Defaults to 'nebius config' parent-id
export NEBIUS_PROJECT_ID=${NEBIUS_PROJECT_ID:-$(nebius config  list | grep parent-id | sed 's/^parent-id: //')}
# echo $NEBIUS_PROJECT_ID

# Looks for the MLFLow Service Account ID, creates it if it doesn't exist
nebius iam service-account list | grep -q "name: mlflow-sa"
if [ $? -eq 0 ]; then
	echo "Using existing mlflow-sa Service Account"
	export NEBIUS_MLFLOW_SA_ID=$(nebius iam service-account list | yq -r '.items[] | select(.metadata.name=="mlflow-sa") | .metadata.id')
else
	echo "Creating mlflow-sa Service Account"
	export NEBIUS_MLFLOW_SA_ID=$(nebius iam service-account create --parent-id ${NEBIUS_PROJECT_ID} --name mlflow-sa --description "Service Account for MLFlow" | yq -r ".metadata.id")
fi

#echo \
nebius msp  mlflow v1alpha1 cluster create  	\
    --parent-id ${NEBIUS_PROJECT_ID} 		\
    --name mlfow-cluster 			\
    --resource-version 2 			\
    --description "MLFlow Tracking Server" 	\
    --public-access false			\
    --admin-username calvin			\
    --admin-password P@ssw0rd			\
    --service-account-id ${NEBIUS_MLFLOW_SA_ID}	\
    --storage-bucket-name mlflow-bucket		\
    --network-id ${NEBIUS_VPC_NETWORK_ID}	# \
    # --size c-2vcpu-8gb 


exit

### README.md

Wrapper around the Nebius MLFlow Managed Service

Expects nebius cli authentiaction and IAM token stored in $NEBIUS_IAM_TOKEN
Calcluates the VPC Network from the first in the list - override with environment variable NEBIUS_VPC_NETWORK_ID

### https://docs.nebius.com/cli/reference/msp/mlflow/v1alpha1/cluster/create

# nebius msp mlflow v1alpha1 cluster create

Creates a cluster.

## Usage

```
nebius msp mlflow v1alpha1 cluster create [data] [flags]
```

## Flags

```
  --async [=<true|false>] (bool) If set, returns operation id. Otherwise, waits for the operation to complete and returns its resource.
  Metadata [required]:                                         Metadata associated with the new cluster. Must include parent_id in which we create the cluster.
    --parent-id <value> (string) [required]                    Identifier of the parent resource to which the resource belongs.
    --name <value> (string)                                    Human readable name for the resource.
    --resource-version <value> (int64)                         Version of the resource for safe concurrent modifications and consistent reads.
                                                               Positive and monotonically increases on each resource spec change (but *not* on each change of the
                                                               resource's container(s) or status).
                                                               Service allows zero value or current.
    --labels <[key1=value1[,key2=value2...]]> (string->string) Labels associated with the resource.
  Spec [required]:                                   Specification for the new cluster.
    --description <value> (string)                   Description of the cluster.
    --public-access [=<true|false>] (bool)           Either make cluster public accessible or accessible only via private VPC.
    --admin-username <value> (string) [required]     MLflow admin username.
    --admin-password <value> (string) [required]     MLflow admin password.
    							password must be between 8 and 64 characters long and contain at least: 
							1 uppercase letter(s), 1 lowercase letter(s), 1 digit(s), 1 special character(s) from -!@#$^&*_=+:;'"\|/?,.`~§±()[]{}<>
    --service-account-id <value> (string) [required] Id of the service account that will be used to access S3 bucket (and create one if not provided).
    --storage-bucket-name <value> (string)           Name of the Nebius S3 bucket for MLflow artifacts. If not provided, will be created under the same parent.
    --network-id <value> (string) [required]         ID of the vpc network.
    --size <value> (string)                          Size defines how much resources will be allocated to mlflow
                                                     See supported sizes in the documentation. Default size is the smallest available in the region.
```


## Global Flags

```
      --color [=<true|false>] (bool)      Enable colored output.
  -c, --config <value> (string)           Provide path to config file.
      --debug [=<true|false>] (bool)      Enable debug logs.
  -f, --file <value> (string)             Input file. For 'update' commands automatically set --full=true.
      --format <value> (string)           Output format. Supported values: json|yaml|table|text.
  -h, --help [=<true|false>] (bool)       Show this message.
      --insecure [=<true|false>] (bool)   Disable transport security.
      --no-browser [=<true|false>] (bool) Do not open browser automatically on auth.
  -p, --profile <value> (string)          Set a profile for interacting with the cloud.
```
