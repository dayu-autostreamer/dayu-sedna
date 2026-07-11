# Sedna Helm Charts

Visit https://github.com/dayu-autostreamer/dayu-sedna for more information.

## Install

```
$ git clone https://github.com/dayu-autostreamer/dayu-sedna.git
$ cd sedna
$ helm install sedna ./build/helm/sedna
```

The defaults intentionally retain the upstream Sedna images for legacy JMES
installations. To use `RuntimeService`, build and publish GM and LC from the
same source revision as this chart, then install with the exact immutable image
references:

```
$ helm install sedna ./build/helm/sedna \
    --set global.runtimeService.enabled=true \
    --set-string global.runtimeService.gmImage=registry.example.com/dayu/sedna-gm:runtime-v1 \
    --set-string global.runtimeService.lcImage=registry.example.com/dayu/sedna-lc:runtime-v1
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
$ controller-gen crd:crdVersions=v1,allowDangerousTypes=true,maxDescLen=0 paths="./pkg/apis/sedna/v1alpha1" output:crd:artifacts:config=build/helm/sedna/crds
```

**NOTE: Set `maxDescLen=0` will generate crd yaml file without description field. Avoid too large data causing helm installation to fail. See [issue](https://github.com/helm/helm/issues/6711).**

The chart ships both `JointMultiEdgeService` and `RuntimeService` CRDs. Install
or upgrade CRDs before rolling out a new GM image so the RuntimeService informer
can complete its initial synchronization. RuntimeService is additive; existing
JMES resources and NodePort deployments remain supported.
