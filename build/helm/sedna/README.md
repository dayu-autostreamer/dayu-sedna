# Sedna Helm Charts

Visit https://github.com/adayangolzz/sedna-modified for more information.

## Install

```
$ git clone https://github.com/adayangolzz/sedna-modified.git
$ cd sedna
$ helm install sedna ./build/helm/sedna
```

## Uninstall

```
$ helm uninstall sedna
```

## Update CRDs

```
$ controller-gen crd:crdVersions=v1,allowDangerousTypes=true,maxDescLen=0 paths="./pkg/apis/sedna/v1alpha1" output:crd:artifacts:config=build/helm/sedna/crds
```

**NOTE: Set `maxDescLen=0` will generate crd yaml file without description field. Avoid too large data causing helm installation to fail. See [issue](https://github.com/helm/helm/issues/6711).**
