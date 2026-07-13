### Deploy All In One Sedna
The [all-in-one script](/scripts/installation/all-in-one.sh) is used to install Sedna along with a mini Kubernetes environment locally, including:
  - A Kubernetes v1.21 cluster with multi worker nodes, default zero worker node.
  - KubeEdge with multi edge nodes, default is latest KubeEdge and one edge node.
  - Sedna, default is the latest version.

It requires you:
  - 2 CPUs or more
  - 2GB+ free memory, depends on node number setting
  - 10GB+ free disk space
  - Internet connection(docker hub, github etc.)
  - Linux platform, such as ubuntu/centos
  - Docker 17.06+

For example: 

  ```bash
  curl https://raw.githubusercontent.com/dayu-autostreamer/dayu-sedna/v1.1/scripts/installation/all-in-one.sh | KUBEEDGE_VERSION=v1.8.0 NUM_EDGE_NODES=2 bash -
  ```

Above command installs a mini Sedna environment, including:
  - A Kubernetes v1.21 cluster with only one master node.
  - KubeEdge with two edge nodes.
  - The latest Sedna.

You can play it online on [katacoda](https://www.katacoda.com/kubeedge-sedna/scenarios/all-in-one).

Advanced options:
| Env Variable |  Description| Default Value|
| --- |  --- | --- |
|NUM_EDGE_NODES     | Number of KubeEdge nodes| 1 |
|NUM_CLOUD_WORKER_NODES    | Number of cloud **worker** nodes, not master node| 0|
|SEDNA_VERSION    | The Sedna version to be installed. |The latest version|
|SEDNA_MANIFEST_REPO | Repository containing install manifests. |dayu-autostreamer/dayu-sedna|
|SEDNA_MANIFEST_REF | Branch, tag, or commit containing install manifests. |v1.1|
|SEDNA_ENABLE_RUNTIME_SERVICE | Fail-fast managed profile; requires explicit GM and LC images. |false|
|SEDNA_GM_IMAGE | Exact GM image; set to an image built from this fork for RuntimeService. |upstream image for `SEDNA_VERSION`|
|SEDNA_LC_IMAGE | Exact LC image; set to an image built from this fork for RuntimeService. |upstream image for `SEDNA_VERSION`|
|SEDNA_KB_IMAGE | Exact KB image override. |upstream image for `SEDNA_VERSION`|
|KUBEEDGE_VERSION    | The KubeEdge version to be installed. |The latest version|
|CLUSTER_NAME       | The all-in-one cluster name| sedna-mini|
|FORCE_INSTALL_SEDNA       | If 'true', force to reinstall Sedna|false|
|NODE_IMAGE       | Custom node image| kubeedge/sedna-allinone-node:v1.21.1|
|REUSE_EDGE_CONTAINER      | Whether reuse edge node containers or not|true|

Clean all-in-one Sedna:  
  ```bash
  curl https://raw.githubusercontent.com/dayu-autostreamer/dayu-sedna/v1.1/scripts/installation/all-in-one.sh | bash /dev/stdin clean
  ```
