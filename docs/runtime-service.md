# RuntimeService managed runtime contract

`RuntimeService` is an additive, revision-scoped API for Dayu runtime workers.
It replaces runtime-side Kubernetes discovery with one control-plane reconciliation
path and one node-local EdgeMesh activation barrier. It does **not** replace or
change `JointMultiEdgeService`; existing JMES/NodePort deployments remain valid.

## Why this is not another Kubernetes cache

The Dayu runtime should not list Pods, Nodes, or NodePort Services. A cache in
every Python process still duplicates watches, has ambiguous negative-cache
semantics, and needs force-refresh calls when lifecycle events race with TTLs.

RuntimeService instead establishes one ownership boundary:

1. the Dayu cloud/backend creates one immutable RuntimeService per worker,
   target node, and deployment revision;
2. the Sedna global manager reconciles one single-replica Deployment and one
   optional ClusterIP Service from shared informer caches;
3. Kubernetes publishes the matching Endpoints object;
4. EdgeMesh validates the exact Service/Endpoints/Pod identity from its existing
   MetaServer-backed informers and programs the userspace portal;
5. the Sedna local controller verifies the node-local EdgeMesh status endpoint
   and acknowledges the exact identity;
6. only then does Sedna report `Activated=True` and `Ready=True`.

Runtime pods have `automountServiceAccountToken: false`. Explicit projected
service-account-token volumes are rejected. They therefore do not need a
Kubernetes client, kubeconfig, API endpoint, or discovery refresh loop.

## Resource and identity invariants

Each RuntimeService owns resources with the same name and namespace:

- one Deployment with `replicas: 1`, `Recreate` strategy, and the exact
  `spec.targetNode`;
- zero or one non-headless ClusterIP Service;
- exactly one named TCP Service port (`runtime`) with a numeric target port;
- exactly one ready Endpoints address, no not-ready address, and a Pod
  `targetRef` matching the unique ready Pod UID.

The controller writes these labels to the Deployment, Pod template, and Service;
Kubernetes copies the Service labels to Endpoints:

| Key | Meaning |
| --- | --- |
| `dayu.io/mesh-managed=true` | opt in to the new EdgeMesh path |
| `dayu.io/install-id` | Dayu installation identity |
| `dayu.io/deployment-revision` | positive base-10 `int64` revision |
| `dayu.io/runtime-id` | RuntimeService name |
| `dayu.io/component` | runtime component |
| `dayu.io/runtime-service-uid` | immutable RuntimeService UID |

The Service also carries `dayu.io/logical-service`, `dayu.io/target-node`, and
the controller-owned `dayu.io/runtime-spec-hash` annotations. The spec hash is
also written to the Deployment/Pod template and closes the failure window where
children were created but the first status update did not reach the API server.
Changing a revision-scoped spec is rejected; create a new RuntimeService.

`spec.podTemplate.metadata.labels` and `annotations` are structural string maps
in both published CRDs. Kubernetes therefore preserves caller metadata and the
controller copies it to the Pod template. The six `dayu.io/*` labels above are
controller-owned: matching caller values are accepted, conflicting values are
rejected, and reconciliation always overlays the authoritative values. Runtime
discovery and metrics should select these guaranteed labels (normally
`dayu.io/mesh-managed=true`, optionally narrowed by install ID or revision),
not an application-specific label.

`status.endpoint.dnsName` is the stable Kubernetes service identity in the form
`<service>.<namespace>.svc.cluster.local`; it intentionally has no terminal
dot. DNS clients running with a high `ndots` value should canonicalize it to an
absolute query name at their connection boundary. The stored status identity
must not vary with client resolver policy.

## Conditions and activation semantics

The status conditions are `SpecAccepted`, `ResourcesReconciled`, `NodeReady`,
`WorkloadReady`, `EndpointReady`, `Activated`, and `Ready`.

`Activated` is a rollout barrier, not a continuous health probe. It means the
target node acknowledged `APPLIED` for the exact tuple:

- RuntimeService UID;
- Service UID;
- endpoint Pod UID;
- deployment revision;
- runtime ID and target node.

The global manager rejects an ACK unless the LC websocket connection declares a
non-empty source node matching the ACK target node. The existing Sedna websocket
transport does not cryptographically authenticate that header; deployments that
cross a trust boundary must secure the GM endpoint separately. A Pod or Service
UID replacement invalidates the tuple and
requires a new acknowledgement. After a successful acknowledgement, normal
runtime health remains the responsibility of Dayu health/telemetry paths.

`Activated` remains true across temporary informer gaps, Pod readiness loss,
or endpoint unavailability when the acknowledged UIDs have not been replaced.
`Ready` remains dynamic and falls whenever node, workload, or endpoint gates no
longer hold. This separation keeps activation as a rollout handshake rather
than a second health-monitoring system.

EdgeMesh `SYNCED` means its primary Service and Endpoints informer caches have
completed initial synchronization. It deliberately does not claim continuous
MetaServer connectivity health.

## Compatibility and rollout order

RuntimeService is additive. The JMES API, generated clients, global controller,
local manager, and NodePort behavior remain registered. EdgeMesh's matching
feature is disabled by default, so upgrading either component alone does not
redirect legacy traffic.

For the managed path, upgrade in this order:

1. install both `runtimeservices.sedna.io` and existing JMES CRDs;
2. apply the updated GM RBAC and deploy GM/LC images built from this same source
   revision (the install script's `SEDNA_ENABLE_RUNTIME_SERVICE=true` profile
   requires `SEDNA_GM_IMAGE` and `SEDNA_LC_IMAGE`; the Helm chart's
   `global.runtimeService.enabled=true` profile requires `gmImage` and
   `lcImage` under the same key);
3. deploy the updated EdgeMesh agent and explicitly enable
   `modules.edgeProxy.managedRuntime.enable: true`;
4. create RuntimeServices only after the node-local EdgeMesh status service is
   ready on `127.0.0.1:10551`.

For an existing installation, update the RuntimeService schema before rolling
out a new GM image. The root installer is a fresh-install workflow, and Helm
does not upgrade CRDs from its `crds/` directory:

```sh
kubectl apply --server-side \
  --field-manager=dayu-sedna-crd-upgrade \
  --force-conflicts \
  -f build/crds/sedna.io_runtimeservices.yaml
```

Server-side apply is intentional because the described raw CRD is too large to
store safely in the client-side last-applied annotation. `--force-conflicts`
transfers schema-field ownership from the original create/install manager; it
does not delete existing RuntimeService objects.

The schema update affects subsequent writes only. It cannot reconstruct
Pod-template labels or annotations that the API server already pruned under
the old schema. Recreate those RuntimeService objects through Dayu's normal
uninstall/install lifecycle (or publish a new runtime revision) when the
caller-supplied metadata itself is required.

Installing the new Sedna binary without the RuntimeService CRD does not remove
legacy controllers, but the new informer will retry until the CRD is installed.
Conversely, installing only the CRD with the default legacy upstream images
does not enable RuntimeService; this is intentional so existing JMES installs
remain backward compatible instead of silently changing their binaries.

See [the sample manifest](../build/crd-samples/sedna/runtimeservice_v1alpha1.yaml).

## Regenerating CRDs

Run `make runtime-service-crds` from the repository root. The target builds the
pinned controller-gen release with a compatible Go toolchain into `_output/tools` and
generates all schemas in an isolated `_output` directory, then publishes only
the RuntimeService raw-install and Helm CRDs. This prevents an API-local change
from reformatting or rewriting unrelated historical Sedna CRDs. The toolchain cap matters because a
controller-gen v0.4.1 binary built with Go 1.22 or newer can fail while loading
types; the repository target selects Go 1.20.14 on newer Go commands while the
project's declared Go 1.17 remains supported. RuntimeService deliberately uses a narrow local pod-template
metadata type instead of embedding Kubernetes `ObjectMeta`; this keeps the wire
shape unchanged while generating only the supported `labels` and `annotations`
maps. Do not replace it with an unrestricted `ObjectMeta`: the pinned generator
would emit an empty nested schema and the API server would prune those maps.
