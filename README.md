# Dayu-Sedna

## Brief Introduction

This project is based on Sedna(https://github.com/kubeedge/sedna).

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
  git clone https://github.com/dayu-autostreamer/dayu-sedna.git
  ```

- install sedna

  ```sh
  curl -LO https://raw.githubusercontent.com/dayu-autostreamer/dayu-sedna/main/scripts/installation/install.sh
  ```

  The default images preserve the upstream Sedna/JMES installation. For the
  managed `RuntimeService` path, first publish GM and LC images built from this
  source revision, then pass their exact references without editing the script:

  ```sh
  SEDNA_ACTION=create \
  SEDNA_ENABLE_RUNTIME_SERVICE=true \
  SEDNA_GM_IMAGE=registry.example.com/dayu/sedna-gm:runtime-v1 \
  SEDNA_LC_IMAGE=registry.example.com/dayu/sedna-lc:runtime-v1 \
  bash install.sh
  ```

  `SEDNA_ENABLE_RUNTIME_SERVICE=true` is a fail-fast install profile: it requires
  both images instead of silently pairing the RuntimeService CRD with old
  binaries. `SEDNA_MANIFEST_REPO` and `SEDNA_MANIFEST_REF` select the matching
  manifests; `SEDNA_KB_IMAGE` is optional. Omitting the profile keeps the legacy
  path operational but does not add RuntimeService support to old binaries.

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
