# Dayu-Sedna

## Brief Introduction

This project is based on Sedna(https://github.com/kubeedge/sedna).

The current Dayu Sedna version is [`v1.1`](https://github.com/dayu-autostreamer/dayu-sedna/tree/v1.1).
The legacy `v1.0` baseline and the managed-runtime changes in `v1.1` are documented in the
[changelog](CHANGELOG.md).

We added the `JointMultiEdgeService` API for the legacy Dayu deployment path and
the additive `RuntimeService` API for revision-scoped, cache-driven managed
runtimes. JMES and its NodePort behavior remain supported.

## Feature

We extended the "jointinferenceservice" section to implement the following features:

- Only cloudWorker or edgeWorker can be deployed separately
- Multiple Edgeworkers can be deployed at once
- Add the file field for backward-compatible file mounting
- Add the mounts field for explicit file/device mounting
- Add the log_level field to match logs
- Keep `DATA_PATH_PREFIX=/home/data` for legacy workers and relative-path default targets
- add ServiceConfig to use nodePort mode for communication

The opt-in RuntimeService path creates one single-replica worker and one optional
ClusterIP endpoint, then waits for an exact node-local EdgeMesh activation ACK.
Runtime pods do not receive a Kubernetes service-account token and no longer
need per-process Pod/Node/Service discovery. See
[the RuntimeService contract](docs/runtime-service.md).



## Quick Start

We assume that you have finished the k8s and kubeedge installation

- git clone

  ```sh
  git clone --branch v1.1 https://github.com/dayu-autostreamer/dayu-sedna.git
  ```

- install sedna

  ```sh
  curl -LO https://raw.githubusercontent.com/dayu-autostreamer/dayu-sedna/v1.1/scripts/installation/install.sh
  ```

  ```sh
  SEDNA_ACTION=create bash install.sh
  ```

- yaml examples

  legacy-compatible file mounting:
  [jointmultiedgeservice_v1alpha1.yaml](build/crd-samples/sedna/jointmultiedgeservice_v1alpha1.yaml)

  explicit file/device mounting:
  [jointmultiedgeservice_mounts_v1alpha1.yaml](build/crd-samples/sedna/jointmultiedgeservice_mounts_v1alpha1.yaml)

  managed runtime example:
  [runtimeservice_v1alpha1.yaml](build/crd-samples/sedna/runtimeservice_v1alpha1.yaml)

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
  kubectl get rts -n <namespace-name>
  ```

## How to Build

clone repository
```bash
git clone --branch v1.1 https://github.com/dayu-autostreamer/dayu-sedna
```

set meta information of building
```bash
# configure buildx buildkitd (default as empty, example at hack/resource/buildkitd_template.toml)
vim hack/resource/buildkitd.toml
# configure buildx driver-opt (default as empty, example at hack/resource/driver_opts_template.toml)
vim hack/resource/driver_opts.toml

# set docker meta info
# default REG is docker.io
# default IMAGE_REPO is $(REG)/dayuhub
# default IMAGE_TAG is v1.1
export REG=xxx
export IMAGE_REPO=xxx
export IMAGE_TAG=v1.1
```

build gm/lc/kb image
  ```sh
  make docker-cross-build
  ```
