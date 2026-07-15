# Sedna Helm Charts

Visit https://github.com/dayu-autostreamer/dayu-sedna for more information.

## Install

```
$ git clone --branch v1.1 https://github.com/dayu-autostreamer/dayu-sedna.git
$ cd dayu-sedna
$ helm install sedna ./build/helm/sedna
```

The defaults intentionally retain the upstream Sedna images for legacy JMES
installations. To use `RuntimeService`, build and publish GM and LC from the
same source revision as this chart, then install with the exact immutable image
references:

```
$ helm install sedna ./build/helm/sedna \
    --set global.runtimeService.enabled=true \
    --set-string global.runtimeService.gmImage=dayuhub/sedna-gm:v1.1 \
    --set-string global.runtimeService.lcImage=dayuhub/sedna-lc:v1.1
```

The managed profile fails rendering unless both explicit images are provided.
The legacy `gm.image` and `lc.image` values remain available when the profile is
disabled; the LC chart now correctly honors `lc.image` instead of hard-coding
it. The KB image can remain upstream because RuntimeService does not change the
knowledge-base component.

## Uninstall

```
$ helm uninstall sedna
```

## Update CRDs

```
$ make runtime-service-crds
$ kubectl apply --server-side \
    --field-manager=dayu-sedna-crd-upgrade \
    --force-conflicts \
    -f build/helm/sedna/crds/sedna.io_runtimeservices.yaml
```

The focused repository target builds a pinned controller-gen with its compatible Go
toolchain and uses RuntimeService's narrow metadata type so pod-template labels
and annotations are not pruned. It publishes only the raw and Helm RuntimeService
manifests, leaving unrelated historical CRDs untouched. The
Helm generation keeps `maxDescLen=0` to avoid oversized release data; see [Helm issue
6711](https://github.com/helm/helm/issues/6711).

[Helm installs but does not upgrade CRDs from `crds/`](https://helm.sh/docs/chart_best_practices/custom_resource_definitions/).
Apply the RuntimeService CRD explicitly before `helm upgrade`; server-side apply
avoids the oversized last-applied annotation produced by client-side apply and
the dedicated field manager makes ownership explicit.

The chart ships both `JointMultiEdgeService` and `RuntimeService` CRDs. Install
or upgrade CRDs before rolling out a new GM image so the RuntimeService informer
can complete its initial synchronization. RuntimeService is additive; existing
JMES resources and NodePort deployments remain supported.
