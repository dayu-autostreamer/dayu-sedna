This guide covers how to install Sedna on an existing Kubernetes environment.

For interested readers, Sedna also has two important components that would be mentioned below, i.e., [GM(GlobalManager)](/README.md#globalmanager) and [LC(LocalController)](/README.md#localcontroller) for workerload generation and maintenance.

If you don't have an existing Kubernetes, you can:
1) Install Kubernetes by following the [Kubernetes website](https://kubernetes.io/docs/setup/).
2) Or follow [quick start](index/quick-start.md) for other options.

### Prerequisites
- [Kubectl][kubectl] with right kubeconfig
- [Kubernetes][kubernetes] 1.16+ cluster running
- [KubeEdge][kubeedge] v1.8+ along with **[EdgeMesh][edgemesh]** running


#### Deploy Sedna

Currently GM is deployed as a [`deployment`][deployment], and LC is deployed as a [`daemonset`][daemonset].


Run the one liner:
```shell
curl https://raw.githubusercontent.com/dayu-autostreamer/dayu-sedna/v1.1/scripts/installation/install.sh | SEDNA_ACTION=create bash -

```

It requires the network to access github since it will download the sedna [crd yamls](/build/crds).
This fork's installer downloads all shipped CRDs, including the legacy
`JointMultiEdgeService` and additive `RuntimeService`, from
`dayu-autostreamer/dayu-sedna`. Set `SEDNA_MANIFEST_REPO` only when using a
compatible manifest mirror; `SEDNA_MANIFEST_REF` selects its branch, tag, or
commit (default `v1.1`).

The default `kubeedge/sedna-*` images preserve existing JMES installations. To
activate RuntimeService, GM and LC must be built from the same fork revision as
the manifests and supplied explicitly:

```shell
curl -LO https://raw.githubusercontent.com/dayu-autostreamer/dayu-sedna/v1.1/scripts/installation/install.sh
SEDNA_ACTION=create \
SEDNA_ENABLE_RUNTIME_SERVICE=true \
SEDNA_GM_IMAGE=dayuhub/sedna-gm:v1.1 \
SEDNA_LC_IMAGE=dayuhub/sedna-lc:v1.1 \
bash install.sh
```

The managed install profile exits before changing the cluster if either GM or
LC image is absent. `SEDNA_KB_IMAGE` can override KB independently;
RuntimeService itself does not require a modified KB.
If you have unstable network to access github or existing sedna source, you can try the way:
```shell
# SEDNA_ROOT is the sedna git source directory or cached directory
export SEDNA_ROOT=/opt/sedna
curl https://raw.githubusercontent.com/dayu-autostreamer/dayu-sedna/v1.1/scripts/installation/install.sh | SEDNA_ACTION=create bash -
```

#### Debug
1\. Check the GM status:
```shell
kubectl get deploy -n sedna gm
```

2\. Check the LC status:
```shell
kubectl get ds lc -n sedna
```

3\. Check the pod status:
```shell
kubectl get pod -n sedna
```

#### Uninstall Sedna
```shell
curl https://raw.githubusercontent.com/dayu-autostreamer/dayu-sedna/v1.1/scripts/installation/install.sh | SEDNA_ACTION=delete bash -
```

[kubectl]:https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/#install-kubectl-binary-with-curl-on-linux
[kubeedge]:https://github.com/kubeedge/kubeedge
[edgemesh]:https://github.com/kubeedge/edgemesh
[kubernetes]:https://kubernetes.io/
[deployment]: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
[daemonset]: https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/
