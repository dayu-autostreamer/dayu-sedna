# Dayu-Sedna

## Brief Introduction

This project is based on Sedna(https://github.com/kubeedge/sedna).

We added the "jointmultiedgeservice" section to complete our functionality.

## Feature

We extended the "jointinferenceservice" section to implement the following features:

- Only cloudWorker or edgeWorker can be deployed separately
- Multiple Edgeworkers can be deployed at once
- Add the file field for backward-compatible file mounting
- Add the mounts field for explicit file/device mounting
- Add worker-level `kubeConfig.path` to follow the upstream `jointmultiedgeservice` kubeconfig style
- Add the log_level field to match logs
- Keep `DATA_PATH_PREFIX=/home/data` for legacy workers and relative-path default targets
- add ServiceConfig to use nodePort mode for communication



## Quick Start

We assume that you have finished the k8s and kubeedge installation

- git clone

  ```sh
  git clone https://github.com/dayu-autostreamer/dayu-sedna.git
  ```

- install sedna

  ```sh
  curl https://raw.githubusercontent.com/kubeedge/sedna/main/scripts/installation/install.sh
  ```

  modify the `TMP_DIR`， `SEDNA_VERSION`  and image(`adayoung/sedna-gm:v0.3.12` and `adayoung/sedna-lc:v0.3.12`)

  ```sh
  SEDNA_ACTION=create bash install.sh
  ```

- yaml examples

  legacy-compatible file mounting:
  [jointmultiedgeservice_v1alpha1.yaml](build/crd-samples/sedna/jointmultiedgeservice_v1alpha1.yaml)

  explicit file/device mounting:
  [jointmultiedgeservice_mounts_v1alpha1.yaml](build/crd-samples/sedna/jointmultiedgeservice_mounts_v1alpha1.yaml)

  worker kubeconfig access:
  - configure it on each worker: `cloudWorker.kubeConfig.path` or `edgeWorker[].kubeConfig.path`
  - the value is the host kubeconfig directory, for example `/home/nvidia/.kube`
  - the controller mounts that directory into the container at `/home/data/.kube`
  - the controller also injects `KUBECONFIG=/home/data/.kube/config`
  - this matches the kubeconfig style used in the upstream `sedna-modified` `jointmultiedgeservice` examples

  when `mounts[].target.path` is omitted:
  - absolute `source.hostPath.path` keeps the same path inside the container
  - `source.hostPath.prefix` only applies when `path` is relative, and the real host source becomes `<prefix>/<path>`
  - relative `source.hostPath.path` is mounted under `DATA_PATH_PREFIX` (`/home/data`)

  ```sh
  kubectl apply -f <yaml-name>
  ```

- get infomation

  ```sh
  kubectl get pod -n <namespace-name>
  kubectl get deploy -n <namespace-name>
  kubectl get svc -n <namespace-name>
  kubectl get mulji -n <namespace-name>
  ```

## How to Build

clone repository
```bash
git clone https://github.com/dayu-autostreamer/dayu-sedna
```

set meta information of building
```bash
# configure buildx buildkitd (default as empty, example at hack/resource/buildkitd_template.toml)
vim hack/resource/buildkitd.toml
# configure buildx driver-opt (default as empty, example at hack/resource/driver_opts_template.toml)
vim hack/resource/driver_opts.toml

# set docker meta info
# default REG is docker.io
# default REPO is dayuhub
# default TAG is v1.0
export REG=xxx
export REPO=xxx
export TAG=xxx
```

build gm/lc/kb image
  ```sh
  make docker-cross-build
  ```
