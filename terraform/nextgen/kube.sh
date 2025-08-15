 1320  helm pull oci://cr.eu-north1.nebius.cloud/marketplace/nebius/ml-flow/chart/mlflow --version 1.0.4
 1335  helm pull oci://cr.eu-north1.nebius.cloud/marketplace/nebius/airflow/chart/airflow --version 1.13.1
 2064  kubectl get pods -A
 2065  kubectl get pods -A | more
 2067  more ~/.nebius/kube_config 
 2069  kubectl get namespaces 
 2070  kubectl get pods -A
 2071  kubectl get pods -A | grep graf
 2072  kubectl get pods -A | grep prom
 2073  kubectl get pods -A | grep db
 2074  kubectl get pods -n monitoring-system
 2075  kubectl get pod metrics-grafana-64cf9479b6-k6f2w
 2083  helm repo add neondatabase https://neondatabase.github.io/helm-charts
 2084  helm install 3.3 oci://registry-1.docker.io/bitnamicharts/mlflow
 2085  helm install oci://registry-1.docker.io/bitnamicharts/mlflow
 2086  helm install mlflow-charts oci://registry-1.docker.io/bitnamicharts/mlflow
 2089  helm repo add spark-operator https://googlecloudplatform.github.io/spark-on-k8s-operator
 2090  helm repo add --force-update spark-operator https://kubeflow.github.io/spark-operator
 2091  helm install spark-operator spark-operator/spark-operator     --namespace spark-operator     --create-namespace     --wait
 2093  wget  https://raw.githubusercontent.com/kubeflow/spark-operator/refs/heads/master/examples/spark-pi.yaml
 2094  kubectl apply -f spark-pi.yaml
 2095  kubectl get sparkapp spark-pi
 2098  kubectl get sparkapp spark-pi



  1312  helm pull oci://cr.eu-north1.nebius.cloud/marketplace/nebius/ml-flow/chart/mlflow --version 1.0.4
 1327  helm pull oci://cr.eu-north1.nebius.cloud/marketplace/nebius/airflow/chart/airflow --version 1.13.1
 2000  kubectl get namespaces
 2001  kubectl get pods -n soperator
 2002  kubectl get pods -n soperator-system
 2003  kubectl get namespaces
 2004  kubectl get namespaces  | grep spark


 
