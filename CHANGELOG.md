# Changelog

All notable changes to the Dayu Sedna fork are documented in this file. The
versions below describe the Dayu integration layer; upstream Sedna API versions
remain unchanged.

## [v1.1] - 2026-07-13

### Added

- Added the `sedna.io/v1alpha1` `RuntimeService` CRD, generated clients, GM and
  LC controllers, and revision-scoped runtime lifecycle management.
- Added exact activation acknowledgement between Sedna LC and Dayu EdgeMesh,
  including observed revision/spec status and exact RuntimeService, Service,
  Pod, and node identities.
- Added managed-runtime CRDs, RBAC, raw-install and Helm profiles, examples,
  operational documentation, and focused controller/manager tests.

### Changed

- Set the default Dayu Sedna image build tag to `v1.1` and pin release install
  examples and manifest downloads to the `v1.1` source tag.
- Bumped the changed Sedna umbrella, GM, and LC Helm charts to chart version
  `0.2.0`; their chart version remains independent from the Dayu release tag.
- Made the managed-runtime install profile fail fast unless matching GM and LC
  images are supplied, preventing new CRDs from being paired with old binaries.

### Compatibility

- `JointMultiEdgeService` and its existing NodePort/file-mount behavior remain
  supported for legacy Dayu deployments.
- The current Dayu managed-runtime path requires both dayu-sedna `v1.1` and
  dayu-edgemesh `v1.1`; `v1.0` does not implement the RuntimeService activation
  contract.

### Fixed

- Fixed the structural RuntimeService CRD schema so Kubernetes preserves
  `spec.podTemplate.metadata.labels` and `annotations` instead of silently
  pruning them before GM reconciliation. Raw-install and Helm CRDs now share
  the same generated metadata contract.
- Narrowed RuntimeService pod-template metadata to the two supported structural
  string maps, pinned its isolated CRD generation toolchain, and added
  regression coverage for the published manifests and controller-enforced
  `dayu.io/*` Pod identity labels.

## [v1.0] - 2026-04-14

- Established the legacy Dayu Sedna baseline based on `JointMultiEdgeService`,
  including Dayu image builds, installer behavior, NodePort communication, and
  file/device mount extensions.

[Unreleased]: https://github.com/dayu-autostreamer/dayu-sedna/compare/v1.1...HEAD
[v1.1]: https://github.com/dayu-autostreamer/dayu-sedna/compare/v1.0...v1.1
[v1.0]: https://github.com/dayu-autostreamer/dayu-sedna/tree/v1.0
