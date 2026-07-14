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


Clone one revision so the installer, CRDs, and RBAC are kept together:
```shell
git clone https://github.com/dayu-autostreamer/dayu-sedna.git
cd dayu-sedna
SEDNA_ACTION=create bash install.sh
```

The root installer reads [CRDs](/build/crds) and GM RBAC from that checkout. It
validates the complete resource set before contacting the cluster and never
downloads individual YAML files. Executing or piping a standalone `install.sh`
is intentionally unsupported because it cannot guarantee that the resources
come from the same revision. For a release, use a complete tagged checkout or
source/release archive and run its root installer.

The default `kubeedge/sedna-*` images preserve existing JMES installations. To
activate RuntimeService, GM and LC must be built from the same fork revision as
the manifests and supplied explicitly:

```shell
SEDNA_ACTION=create \
SEDNA_ENABLE_RUNTIME_SERVICE=true \
SEDNA_GM_IMAGE=dayuhub/sedna-gm:v1.1 \
SEDNA_LC_IMAGE=dayuhub/sedna-lc:v1.1 \
bash install.sh
```

The managed install profile exits before changing the cluster if either GM or
LC image is absent. `SEDNA_KB_IMAGE` can override KB independently;
RuntimeService itself does not require a modified KB.

For an offline installation, copy or extract the complete checkout/archive on
the target host. `SEDNA_ROOT` may point to another directory only when that
directory contains the complete version-matched `build/crds` and
`build/gm/rbac` resource tree.

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
cd dayu-sedna
SEDNA_ACTION=delete bash install.sh
```

[kubectl]:https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/#install-kubectl-binary-with-curl-on-linux
[kubeedge]:https://github.com/kubeedge/kubeedge
[edgemesh]:https://github.com/kubeedge/edgemesh
[kubernetes]:https://kubernetes.io/
[deployment]: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
[daemonset]: https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/
