#!/usr/bin/env bash

# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Influential env vars:
#
# SEDNA_ACTION    | optional | 'create'/'delete', default is 'create'
# SEDNA_ROOT      | optional | Directory containing install.sh and build resources.
#                              Defaults to this script's repository/bundle root.
# REG             | optional | Registry hosting the dayuhub images, default 'docker.io'.
# SEDNA_IMAGE_REPO | optional | Complete Dayu-Sedna image repository.
#                              Defaults to '<REG>/dayuhub'.
# SEDNA_GM_IMAGE  | optional | Exact Dayu-Sedna GM image override.
# SEDNA_LC_IMAGE  | optional | Exact Dayu-Sedna LC image override.
# SEDNA_KB_IMAGE  | optional | Exact KB image.

# This repository-root file is the complete, canonical installer. Keep
# compatibility entrypoints and all-in-one flows pointed here instead of
# maintaining or downloading another installer implementation.

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_SOURCE=${BASH_SOURCE[0]:-}
if [ -z "$SCRIPT_SOURCE" ] || [ "$SCRIPT_SOURCE" = "-" ] || [ ! -f "$SCRIPT_SOURCE" ]; then
  echo "install.sh must be executed from a Dayu-Sedna checkout or release bundle." >&2
  echo "Downloading or piping the script alone is not supported because the version-matched CRD and RBAC resources are required." >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)
TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/sedna.XXXXXX")
SEDNA_ROOT=${SEDNA_ROOT:-$SCRIPT_DIR}
DAYU_SEDNA_VERSION=v1.1
REQUESTED_SEDNA_VERSION=${SEDNA_VERSION:-}
SEDNA_VERSION=$DAYU_SEDNA_VERSION
if [ -n "$REQUESTED_SEDNA_VERSION" ] && [ "v${REQUESTED_SEDNA_VERSION#v}" != "$DAYU_SEDNA_VERSION" ]; then
  echo "Ignoring SEDNA_VERSION=$REQUESTED_SEDNA_VERSION; this installer is pinned to Dayu-Sedna $DAYU_SEDNA_VERSION." >&2
fi
REGISTRY=${REG:-docker.io}
SEDNA_IMAGE_REPO=${SEDNA_IMAGE_REPO:-${REGISTRY%/}/dayuhub}
SEDNA_IMAGE_REPO=${SEDNA_IMAGE_REPO%/}
SEDNA_KB_IMAGE=${SEDNA_KB_IMAGE:-$SEDNA_IMAGE_REPO/sedna-kb:$SEDNA_VERSION}
SEDNA_GM_IMAGE=${SEDNA_GM_IMAGE:-$SEDNA_IMAGE_REPO/sedna-gm:$SEDNA_VERSION}
SEDNA_LC_IMAGE=${SEDNA_LC_IMAGE:-$SEDNA_IMAGE_REPO/sedna-lc:$SEDNA_VERSION}


trap "rm -rf '$TMP_DIR'" EXIT

required_install_resources=(
  build/crds/sedna.io_datasets.yaml
  build/crds/sedna.io_featureextractionservices.yaml
  build/crds/sedna.io_federatedlearningjobs.yaml
  build/crds/sedna.io_incrementallearningjobs.yaml
  build/crds/sedna.io_jointinferenceservices.yaml
  build/crds/sedna.io_jointmultiedgeservices.yaml
  build/crds/sedna.io_lifelonglearningjobs.yaml
  build/crds/sedna.io_models.yaml
  build/crds/sedna.io_objectsearchservices.yaml
  build/crds/sedna.io_objecttrackingservices.yaml
  build/crds/sedna.io_reidjobs.yaml
  build/crds/sedna.io_runtimeservices.yaml
  build/crds/sedna.io_videoanalyticsjobs.yaml
  build/gm/rbac/gm.yaml
)

validate_install_resources() {
  local resource missing=0
  if [ ! -d "$SEDNA_ROOT" ]; then
    echo "install resource root does not exist: $SEDNA_ROOT" >&2
    exit 2
  fi
  SEDNA_ROOT=$(cd "$SEDNA_ROOT" && pwd)

  for resource in "${required_install_resources[@]}"; do
    if [ ! -f "$SEDNA_ROOT/$resource" ]; then
      echo "missing install resource: $SEDNA_ROOT/$resource" >&2
      missing=1
    fi
  done

  if [ "$missing" -ne 0 ]; then
    echo "Use a complete Dayu-Sedna checkout or release bundle from one revision." >&2
    exit 2
  fi
}

prepare_install(){
  # need to create the namespace first
  kubectl create ns sedna
}

cleanup(){
  kubectl delete ns sedna
}

create_crds() {
  cd "$SEDNA_ROOT"
  kubectl create -f build/crds
}

delete_crds() {
  cd "$SEDNA_ROOT"
  kubectl delete -f build/crds --timeout=90s
}

get_service_address() {
  local service=$1
  local port=$(kubectl -n sedna get svc $service -ojsonpath='{.spec.ports[0].port}')

  # <service-name>.<namespace>:<port>
  echo $service.sedna:$port
}

create_kb(){
  cd "$SEDNA_ROOT"

  kubectl $action -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: kb
  namespace: sedna
spec:
  selector:
    sedna: kb
  type: ClusterIP
  ports:
    - protocol: TCP
      port: 9020
      targetPort: 9020
      name: "tcp-0"  # required by edgemesh, to clean
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kb
  labels:
    sedna: kb
  namespace: sedna
spec:
  replicas: 1
  selector:
    matchLabels:
      sedna: kb
  template:
    metadata:
      labels:
        sedna: kb
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: node-role.kubernetes.io/edge
                operator: DoesNotExist
      serviceAccountName: sedna
      containers:
      - name: kb
        imagePullPolicy: IfNotPresent
        image: $SEDNA_KB_IMAGE
        env:
          - name: KB_URL
            value: "sqlite:///db/kb.sqlite3"
        volumeMounts:
        - name: kb-url
          mountPath: /db
        resources:
          requests:
            memory: 256Mi
            cpu: 100m
          limits:
            memory: 512Mi
      volumes:
        - name: kb-url
          hostPath:
            path: /opt/kb-data
            type: DirectoryOrCreate
EOF
}

prepare_gm_config_map() {

  KB_ADDRESS=$(get_service_address kb)

  cm_name=${1:-gm-config}
  config_file=${TMP_DIR}/${2:-gm.yaml}

  if [ -n "${SEDNA_GM_CONFIG:-}" ] && [ -f "${SEDNA_GM_CONFIG}" ] ; then
    cp "$SEDNA_GM_CONFIG" $config_file
  else
    cat > $config_file << EOF
kubeConfig: ""
master: ""
namespace: ""
websocket:
  address: 0.0.0.0
  port: 9000
localController:
  server: http://localhost:${SEDNA_LC_BIND_PORT:-9100}
knowledgeBaseServer:
  server: http://$KB_ADDRESS
EOF
  fi

  kubectl $action -n sedna configmap $cm_name --from-file=$config_file
}

create_gm() {
  cd "$SEDNA_ROOT"

  kubectl create -f build/gm/rbac/

  cm_name=gm-config
  config_file_name=gm.yaml
  prepare_gm_config_map $cm_name $config_file_name


  kubectl $action -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: gm
  namespace: sedna
spec:
  selector:
    sedna: gm
  type: ClusterIP
  ports:
    - protocol: TCP
      port: 9000
      targetPort: 9000
      name: "tcp-0"  # required by edgemesh, to clean
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gm
  labels:
    sedna: gm
  namespace: sedna
spec:
  replicas: 1
  selector:
    matchLabels:
      sedna: gm
  template:
    metadata:
      labels:
        sedna: gm
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: node-role.kubernetes.io/edge
                operator: DoesNotExist
      serviceAccountName: sedna
      containers:
      - name: gm
        imagePullPolicy: IfNotPresent
        image: $SEDNA_GM_IMAGE
        command: ["sedna-gm", "--config", "/config/$config_file_name", "-v2"]
        volumeMounts:
        - name: gm-config
          mountPath: /config
        resources:
          requests:
            memory: 32Mi
            cpu: 100m
          limits:
            memory: 256Mi
      volumes:
        - name: gm-config
          configMap:
            name: $cm_name
EOF
}

delete_gm() {
  cd "$SEDNA_ROOT"

  kubectl delete -f build/gm/rbac/

  # no need to clean gm deployment alone
}

create_lc() {

  GM_ADDRESS=$(get_service_address gm)

  kubectl $action -f- <<EOF
apiVersion: apps/v1
kind: DaemonSet
metadata:
  labels:
    sedna: lc
  name: lc
  namespace: sedna
spec:
  selector:
    matchLabels:
      sedna: lc
  template:
    metadata:
      labels:
        sedna: lc
    spec:
      containers:
        - name: lc
          imagePullPolicy: IfNotPresent
          image: $SEDNA_LC_IMAGE
          env:
            - name: GM_ADDRESS
              value: $GM_ADDRESS
            - name: BIND_PORT
              value: "${SEDNA_LC_BIND_PORT:-9100}"
            - name: NODENAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName
            - name: ROOTFS_MOUNT_DIR
              # the value of ROOTFS_MOUNT_DIR is same with the mount path of volume
              value: /rootfs
          resources:
            requests:
              memory: 32Mi
              cpu: 100m
            limits:
              memory: 128Mi
          volumeMounts:
            - name: localcontroller
              mountPath: /rootfs
      volumes:
        - name: localcontroller
          hostPath:
            path: /
      restartPolicy: Always
      hostNetwork: true
      dnsPolicy: ClusterFirstWithHostNet
EOF
}

delete_lc() {
  # ns would be deleted in delete_gm
  # so no need to clean lc alone
  return
}

wait_ok() {
  echo "Waiting control components to be ready..."
  kubectl -n sedna wait --for=condition=available --timeout=600s deployment/gm
  kubectl -n sedna wait pod --for=condition=Ready --selector=sedna
  kubectl -n sedna get pod
}

delete_pods() {
  # in case some nodes are not ready, here delete with a 60s timeout, otherwise force delete these
  kubectl -n sedna delete pod --all --timeout=60s || kubectl -n sedna delete pod --all --force --grace-period=0
}

check_kubectl () {
  kubectl get pod >/dev/null
}

check_action() {
  action=${SEDNA_ACTION:-create}
  support_action_list="create delete"
  if ! echo "$support_action_list" | grep -w -q "$action"; then
    echo "\`$action\` not in support action list: create/delete!" >&2
    echo "You need to specify it by setting $(red_text SEDNA_ACTION) environment variable when running this script!" >&2
    exit 2
  fi

}

check_dayu_images() {
  if [ "$action" != create ]; then
    return 0
  fi

  local component image_variable image
  for component in KB GM LC; do
    image_variable="SEDNA_${component}_IMAGE"
    image=${!image_variable}
    if [ -z "$image" ]; then
      echo "$image_variable must not be empty." >&2
      exit 2
    fi
  done
}

do_check() {
  check_action
  check_dayu_images
  check_kubectl
}

show_debug_infos() {
  cat - <<EOF
Dayu-Sedna $SEDNA_VERSION is $(green_text running):
See GM status: kubectl -n sedna get deploy
See LC status: kubectl -n sedna get ds lc
See Pod status: kubectl -n sedna get pod
EOF
}

NO_COLOR='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
green_text() {
  echo -ne "$GREEN$@$NO_COLOR"
}

red_text() {
  echo -ne "$RED$@$NO_COLOR"
}

validate_install_resources
do_check

case "$action" in
  create)
    echo "Installing Dayu-Sedna $SEDNA_VERSION..."
    echo "  KB image: $SEDNA_KB_IMAGE"
    echo "  GM image: $SEDNA_GM_IMAGE"
    echo "  LC image: $SEDNA_LC_IMAGE"
    prepare_install
    create_crds
    create_kb
    create_gm
    create_lc
    wait_ok
    show_debug_infos
    ;;

  delete)
    # no errexit when fail to clean
    set +o errexit
    delete_pods
    delete_gm
    delete_lc
    delete_crds
    cleanup
    echo "$(green_text Dayu-Sedna is uninstalled successfully)"
    ;;
esac
