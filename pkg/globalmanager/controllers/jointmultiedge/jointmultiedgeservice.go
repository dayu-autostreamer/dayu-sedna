/*
Copyright 2021 The KubeEdge Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package jointmultiedge

import (
	"context"
	"fmt"
	"time"
	"path/filepath"
	"reflect"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	k8scontroller "k8s.io/kubernetes/pkg/controller"

	sednav1 "github.com/adayangolzz/sedna-modified/pkg/apis/sedna/v1alpha1"
	sednaclientset "github.com/adayangolzz/sedna-modified/pkg/client/clientset/versioned/typed/sedna/v1alpha1"
	sednav1listers "github.com/adayangolzz/sedna-modified/pkg/client/listers/sedna/v1alpha1"
	"github.com/adayangolzz/sedna-modified/pkg/globalmanager/config"
	"github.com/adayangolzz/sedna-modified/pkg/globalmanager/runtime"
)

const (
	// Name is this controller name
	Name = "JointMultiEdge"

	// KindName is the kind name of CR this controller controls
	KindName = "JointMultiEdgeService"
)

const (
	jointMultiEdgeForEdge  = "Edge"
	jointMultiEdgeForCloud = "Cloud"
)

// Kind contains the schema.GroupVersionKind for this controller type.
var Kind = sednav1.SchemeGroupVersion.WithKind(Name)

// Controller ensures that all JointMultiEdgeService objects
// have corresponding pods to run their configured workload.
type Controller struct {
	kubeClient kubernetes.Interface
	client     sednaclientset.SednaV1alpha1Interface

	// podStoreSynced returns true if the pod store has been synced at least once.
	podStoreSynced cache.InformerSynced
	// A store of pods
	podStore corelisters.PodLister

	// serviceStoreSynced returns true if the JointMultiEdgeService store has been synced at least once.
	serviceStoreSynced cache.InformerSynced
	// A store of service
	serviceLister sednav1listers.JointMultiEdgeServiceLister

	// JointMultiEdgeServices that need to be updated
	queue workqueue.RateLimitingInterface

	recorder record.EventRecorder

	cfg *config.ControllerConfig

	sendToEdgeFunc runtime.DownstreamSendFunc
}

// Run starts the main goroutine responsible for watching and syncing services.
func (c *Controller) Run(stopCh <-chan struct{}) {
	workers := 1

	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Infof("Starting %s controller", Name)
	defer klog.Infof("Shutting down %s controller", Name)

	if !cache.WaitForNamedCacheSync(Name, stopCh, c.podStoreSynced, c.serviceStoreSynced) {
		klog.Errorf("failed to wait for %s caches to sync", Name)

		return
	}

	klog.Infof("Starting %s workers", Name)
	for i := 0; i < workers; i++ {
		go wait.Until(c.worker, time.Second, stopCh)
	}

	<-stopCh
}

// enqueueByPod enqueues the JointMultiEdgeService object of the specified pod.
func (c *Controller) enqueueByPod(pod *v1.Pod, immediate bool) {
	controllerRef := metav1.GetControllerOf(pod)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != Kind.Kind {
		return
	}

	service, err := c.serviceLister.JointMultiEdgeServices(pod.Namespace).Get(controllerRef.Name)
	if err != nil {
		return
	}

	if service.UID != controllerRef.UID {
		return
	}

	c.enqueueController(service, immediate)
}

// When a pod is created, enqueue the controller that manages it and update it's expectations.
func (c *Controller) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)
	if pod.DeletionTimestamp != nil {
		// on a restart of the controller, it's possible a new pod shows up in a state that
		// is already pending deletion. Prevent the pod from being a creation observation.
		c.deletePod(pod)
		return
	}

	// backoff to queue when PodFailed
	immediate := pod.Status.Phase != v1.PodFailed

	c.enqueueByPod(pod, immediate)
}

// When a pod is updated, figure out what joint multiedge service manage it and wake them up.
func (c *Controller) updatePod(old, cur interface{}) {
	curPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)

	// no pod update, no queue
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		return
	}

	c.addPod(curPod)
}

// deletePod enqueues the JointmultiedgeService obj When a pod is deleted
func (c *Controller) deletePod(obj interface{}) {
	pod, ok := obj.(*v1.Pod)

	// comment from https://github.com/kubernetes/kubernetes/blob/master/pkg/controller/job/job_controller.go

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new JointMultiEdgeService will not be woken up till the periodic resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			klog.Warningf("couldn't get object from tombstone %+v", obj)
			return
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			klog.Warningf("tombstone contained object that is not a pod %+v", obj)
			return
		}
	}
	c.enqueueByPod(pod, true)
}

// obj could be an *sednav1.JointMultiEdgeService, or a DeletionFinalStateUnknown marker item,
// immediate tells the controller to update the status right away, and should
// happen ONLY when there was a successful pod run.
func (c *Controller) enqueueController(obj interface{}, immediate bool) {
	key, err := k8scontroller.KeyFunc(obj)
	if err != nil {
		klog.Warningf("Couldn't get key for object %+v: %v", obj, err)
		return
	}

	backoff := time.Duration(0)
	if !immediate {
		backoff = runtime.GetBackoff(c.queue, key)
	}
	c.queue.AddAfter(key, backoff)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the sync is never invoked concurrently with the same key.
func (c *Controller) worker() {
	for c.processNextWorkItem() {
	}
}

func (c *Controller) processNextWorkItem() bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	forget, err := c.sync(key.(string))
	if err == nil {
		if forget {
			c.queue.Forget(key)
		}
		return true
	}

	klog.Warningf("Error syncing jointmultiedge service: %v", err)
	c.queue.AddRateLimited(key)

	return true
}

// sync will sync the jointmultiedgeservice with the given key.
// This function is not meant to be invoked concurrently with the same key.
func (c *Controller) sync(key string) (bool, error) {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing jointmultiedge service %q (%v)", key, time.Since(startTime))
	}()

	ns, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return false, err
	}
	if len(ns) == 0 || len(name) == 0 {
		return false, fmt.Errorf("invalid jointmultiedge service key %q: either namespace or name is missing", key)
	}
	sharedService, err := c.serviceLister.JointMultiEdgeServices(ns).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			klog.V(4).Infof("JointMultiEdgeService has been deleted: %v", key)
			return true, nil
		}
		return false, err
	}

	service := *sharedService

	// if service was finished previously, we don't want to redo the termination
	if isServiceFinished(&service) {
		return true, nil
	}

	// set kind for service in case that the kind is None
	// more details at https://github.com/kubernetes/kubernetes/issues/3030
	service.SetGroupVersionKind(Kind)

	selector, _ := runtime.GenerateSelector(&service)
	pods, err := c.podStore.Pods(service.Namespace).List(selector)

	if err != nil {
		return false, err
	}

	klog.V(4).Infof("list jointmultiedge service %v/%v, %v pods: %v", service.Namespace, service.Name, len(pods), pods)

	latestConditionLen := len(service.Status.Conditions)

	active := runtime.CalcActivePodCount(pods)
	var failed int32 = 0

	// neededCounts means that two pods should be created successfully in a jointmultiedge service currently
	// two pods consist of edge pod and cloud pod
	var neededCounts int32 = 2

	if service.Status.StartTime == nil {
		now := metav1.Now()
		service.Status.StartTime = &now
	} else {
		failed = neededCounts - active
	}

	var manageServiceErr error
	serviceFailed := false

	var latestConditionType sednav1.JointMultiEdgeServiceConditionType = ""

	// get the latest condition type
	// based on that condition updated is appended, not inserted.
	jobConditions := service.Status.Conditions
	if len(jobConditions) > 0 {
		latestConditionType = (jobConditions)[len(jobConditions)-1].Type
	}

	var newCondtionType sednav1.JointMultiEdgeServiceConditionType
	var reason string
	var message string

	if failed > 0 {
		serviceFailed = true
		// TODO: get the failed worker, and knows that which worker fails, edge multiedge worker or cloud multiedge worker
		reason = "workerFailed"
		message = "the worker of service failed"
		newCondtionType = sednav1.JointMultiEdgeServiceCondFailed
		c.recorder.Event(&service, v1.EventTypeWarning, reason, message)
	} else {
		if len(pods) == 0 {
			active, manageServiceErr = c.createWorkers(&service)
		}
		if manageServiceErr != nil {
			serviceFailed = true
			message = error.Error(manageServiceErr)
			newCondtionType = sednav1.JointMultiEdgeServiceCondFailed
			failed = neededCounts - active
		} else {
			// TODO: handle the case that the pod phase is PodSucceeded
			newCondtionType = sednav1.JointMultiEdgeServiceCondRunning
		}
	}

	//
	if newCondtionType != latestConditionType {
		service.Status.Conditions = append(service.Status.Conditions, newServiceCondition(newCondtionType, reason, message))
	}
	forget := false

	// no need to update the jointmultiedgeservice if the status hasn't changed since last time
	if service.Status.Active != active || service.Status.Failed != failed || len(service.Status.Conditions) != latestConditionLen {
		service.Status.Active = active
		service.Status.Failed = failed

		if err := c.updateStatus(&service); err != nil {
			return forget, err
		}

		if serviceFailed && !isServiceFinished(&service) {
			// returning an error will re-enqueue jointmultiedgeservice after the backoff period
			return forget, fmt.Errorf("failed pod(s) detected for jointmultiedge service key %q", key)
		}

		forget = true
	}

	return forget, manageServiceErr
}

// newServiceCondition creates a new joint condition
func newServiceCondition(conditionType sednav1.JointMultiEdgeServiceConditionType, reason, message string) sednav1.JointMultiEdgeServiceCondition {
	return sednav1.JointMultiEdgeServiceCondition{
		Type:               conditionType,
		Status:             v1.ConditionTrue,
		LastHeartbeatTime:  metav1.Now(),
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	}
}

func (c *Controller) updateStatus(service *sednav1.JointMultiEdgeService) error {
	client := c.client.JointMultiEdgeServices(service.Namespace)
	return runtime.RetryUpdateStatus(service.Name, service.Namespace, func() error {
		newService, err := client.Get(context.TODO(), service.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		newService.Status = service.Status
		_, err = client.UpdateStatus(context.TODO(), newService, metav1.UpdateOptions{})
		return err
	})
}

func isServiceFinished(j *sednav1.JointMultiEdgeService) bool {
	for _, c := range j.Status.Conditions {
		if (c.Type == sednav1.JointMultiEdgeServiceCondFailed) && c.Status == v1.ConditionTrue {
			return true
		}
	}
	return false
}

func (c *Controller) createWorkers(service *sednav1.JointMultiEdgeService) (active int32, err error) {
	active = 0

	// create cloud worker
	err = c.createCloudWorker(service)
	if err != nil {
		return active, err
	}
	active++

	// create k8s service for cloudPod
	// bigModelHost, err := runtime.CreateEdgeMeshService(c.kubeClient, service, jointMultiEdgeForCloud, bigModelPort)
	bigModelHost, err := runtime.CreateEdgeMeshServiceCustome(c.kubeClient, service)
	if err != nil {
		return active, err
	}

	// create edge worker
	err = c.createEdgeWorker(service, bigModelHost)
	if err != nil {
		return active, err
	}
	active++

	return active, err
}



func (c *Controller) createCloudWorker(service *sednav1.JointMultiEdgeService) error {
    if reflect.DeepEqual(service.Spec.CloudWorker, sednav1.CloudWorker{}) {
        return nil
    }

    cloudWorker := service.Spec.CloudWorker
    var workerParam = runtime.WorkerParam{
		Env:    make(map[string]string),
		Mounts: make([]runtime.WorkerMount, 0),
	}
    logLevel := service.Spec.CloudWorker.LogLevel.Level

    
    envMap := make(map[string]string)
    mountedPaths := make(map[string]struct{})
    volumeCounter := 0 

	workerParam.Env = map[string]string{
        "NAMESPACE":          service.Namespace,
        "SERVICE_NAME":       service.Name,
        "LOG_LEVEL":         logLevel,
        "NODE_NAME":         service.Spec.CloudWorker.Template.Spec.NodeName,
        "DATA_PATH_PREFIX":  "/home/data",
    }

    // multiple file mount
    for _, path := range service.Spec.CloudWorker.File.Paths {
        dirPath := filepath.Dir(path)

		// not exist
        if _, exists := mountedPaths[dirPath]; !exists {
            mountedPaths[dirPath] = struct{}{}

            volumeName := fmt.Sprintf("volume%d", volumeCounter)

            envMap[volumeName] = dirPath

            // 添加挂载配置
            workerParam.Mounts = append(workerParam.Mounts, runtime.WorkerMount{
                URL: &runtime.MountURL{
                    URL:                   dirPath,
                    DownloadByInitializer: true,
                },
                Name: volumeName,
                EnvName: volumeName,
            })

            workerParam.Env[fmt.Sprintf("VOLUME_%d", volumeCounter)] = dirPath
            volumeCounter++
        }
    }

    

    
	workerParam.Env["VOLUME_NUM"] = fmt.Sprintf("%d", volumeCounter)

	if len(cloudWorker.Template.Spec.Containers) == 0 {
		return fmt.Errorf("containers in cloud worker template is empty")
	}

    for i := range cloudWorker.Template.Spec.Containers {
        container := &cloudWorker.Template.Spec.Containers[i]

        // inject env
        for key, value := range workerParam.Env {
            container.Env = append(container.Env, v1.EnvVar{
                Name:  key,
                Value: value,
            })
        }

        // mount file
        for volumeName, _ := range envMap {
            container.VolumeMounts = append(container.VolumeMounts, v1.VolumeMount{
                Name:      volumeName,
                MountPath: fmt.Sprintf("%s/%s", workerParam.Env["DATA_PATH_PREFIX"], volumeName),
            })
        }
    }

	if cloudWorker.Template.ObjectMeta.Labels == nil {
		cloudWorker.Template.ObjectMeta.Labels = make(map[string]string)
	}
	
	cloudWorker.Template.ObjectMeta.Labels["kubernetes.io/hostname"] = cloudWorker.Template.Spec.NodeName
	cloudWorker.Template.ObjectMeta.Labels["jointmultiedge.sedna.io/name"] = service.Name


    deployment := &appsv1.Deployment{
        ObjectMeta: metav1.ObjectMeta{
            Name:      service.Name + "-cloudworker-" + utilrand.String(5),
            Namespace: service.Namespace,
        },
        Spec: appsv1.DeploymentSpec{
            Replicas: int32Ptr(1),
            Selector: &metav1.LabelSelector{
                MatchLabels: map[string]string{
                    "kubernetes.io/hostname": cloudWorker.Template.Spec.NodeName,
                    "jointmultiedge.sedna.io/name": service.Name,
                },
            },
            Template: cloudWorker.Template,
        },
    }

	deployment.Spec.Template.Spec.Volumes = make([]v1.Volume, 0)

    for volumeName, dirPath := range envMap {
        deployment.Spec.Template.Spec.Volumes = append(deployment.Spec.Template.Spec.Volumes, v1.Volume{
            Name: volumeName,
            VolumeSource: v1.VolumeSource{
                HostPath: &v1.HostPathVolumeSource{
                    Path: dirPath,
                },
            },
        })
    }

    _, err := c.kubeClient.AppsV1().Deployments(service.Namespace).Create(context.TODO(), deployment, metav1.CreateOptions{})
    if err != nil {
        return err
    }

    return nil
}


func (c *Controller) createEdgeWorker(service *sednav1.JointMultiEdgeService, bigModelHost string) error {
    if reflect.DeepEqual(service.Spec.EdgeWorker, sednav1.EdgeWorker{}) {
        return nil
    }

    for _, edgeWorker := range service.Spec.EdgeWorker {
        var workerParam = runtime.WorkerParam{
            Env:    make(map[string]string),
            Mounts: make([]runtime.WorkerMount, 0),
        }
        logLevel := edgeWorker.LogLevel.Level

        envMap := make(map[string]string)
        mountedPaths := make(map[string]struct{})
        volumeCounter := 0

		workerParam.Env = map[string]string{
            "NAMESPACE":          service.Namespace,
            "SERVICE_NAME":       service.Name,
            "LOG_LEVEL":         logLevel,
            "NODE_NAME":         edgeWorker.Template.Spec.NodeName,
            "DATA_PATH_PREFIX":  "/home/data",
            "LC_SERVER":         c.cfg.LC.Server,
        }
		
        // multiple file mount
        for _, path := range edgeWorker.File.Paths {
            dirPath := filepath.Dir(path)

            // not exist
            if _, exists := mountedPaths[dirPath]; !exists {
                mountedPaths[dirPath] = struct{}{}

                volumeName := fmt.Sprintf("volume%d", volumeCounter)

                envMap[volumeName] = dirPath

                // 添加挂载配置
                workerParam.Mounts = append(workerParam.Mounts, runtime.WorkerMount{
                    URL: &runtime.MountURL{
                        URL:                   dirPath,
                        DownloadByInitializer: true,
                    },
                    Name: volumeName,
                    EnvName: volumeName,
                })

                workerParam.Env[fmt.Sprintf("VOLUME_%d", volumeCounter)] = dirPath
                volumeCounter++
            }
        }

        

        
		workerParam.Env["VOLUME_NUM"] = fmt.Sprintf("%d", volumeCounter)

        if len(edgeWorker.Template.Spec.Containers) == 0 {
            return fmt.Errorf("containers in edge worker template is empty")
        }

        for i := range edgeWorker.Template.Spec.Containers {
            container := &edgeWorker.Template.Spec.Containers[i]

            // inject env
            for key, value := range workerParam.Env {
                container.Env = append(container.Env, v1.EnvVar{
                    Name:  key,
                    Value: value,
                })
            }

            // mount file
            for volumeName, _ := range envMap {
                container.VolumeMounts = append(container.VolumeMounts, v1.VolumeMount{
                    Name:      volumeName,
                    MountPath: fmt.Sprintf("%s/%s", workerParam.Env["DATA_PATH_PREFIX"], volumeName),
                })
            }
        }

        if edgeWorker.Template.ObjectMeta.Labels == nil {
            edgeWorker.Template.ObjectMeta.Labels = make(map[string]string)
        }
        edgeWorker.Template.ObjectMeta.Labels["kubernetes.io/hostname"] = edgeWorker.Template.Spec.NodeName
        edgeWorker.Template.ObjectMeta.Labels["jointmultiedge.sedna.io/name"] = service.Name

        deployment := &appsv1.Deployment{
            ObjectMeta: metav1.ObjectMeta{
                Name:      service.Name + "-edgeworker-" + utilrand.String(5),
                Namespace: service.Namespace,
            },
            Spec: appsv1.DeploymentSpec{
                Replicas: int32Ptr(1),
                Selector: &metav1.LabelSelector{
                    MatchLabels: map[string]string{
                        "kubernetes.io/hostname": edgeWorker.Template.Spec.NodeName,
                        "jointmultiedge.sedna.io/name": service.Name,
                    },
                },
                Template: edgeWorker.Template,
            },
        }

        deployment.Spec.Template.Spec.Volumes = make([]v1.Volume, 0)

        for volumeName, dirPath := range envMap {
            deployment.Spec.Template.Spec.Volumes = append(deployment.Spec.Template.Spec.Volumes, v1.Volume{
                Name: volumeName,
                VolumeSource: v1.VolumeSource{
                    HostPath: &v1.HostPathVolumeSource{
                        Path: dirPath,
                    },
                },
            })
        }

        _, err := c.kubeClient.AppsV1().Deployments(service.Namespace).Create(context.TODO(), deployment, metav1.CreateOptions{})
        if err != nil {
            return err
        }
    }

    return nil
}

func int32Ptr(i int32) *int32 {
    return &i
}

// New creates a new JointMultiEdgeService controller that keeps the relevant pods
// in sync with their corresponding JointMultiEdgeService objects.
func New(cc *runtime.ControllerContext) (runtime.FeatureControllerI, error) {
	cfg := cc.Config

	podInformer := cc.KubeInformerFactory.Core().V1().Pods()

	serviceInformer := cc.SednaInformerFactory.Sedna().V1alpha1().JointMultiEdgeServices()

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: cc.KubeClient.CoreV1().Events("")})

	jc := &Controller{
		kubeClient: cc.KubeClient,
		client:     cc.SednaClient.SednaV1alpha1(),

		queue:    workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(runtime.DefaultBackOff, runtime.MaxBackOff), "jointmultiedgeservice"),
		recorder: eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "jointmultiedgeservice-controller"}),
		cfg:      cfg,
	}

	serviceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			jc.enqueueController(obj, true)
			jc.syncToEdge(watch.Added, obj)
		},

		UpdateFunc: func(old, cur interface{}) {
			jc.enqueueController(cur, true)
			jc.syncToEdge(watch.Added, cur)
		},

		DeleteFunc: func(obj interface{}) {
			jc.enqueueController(obj, true)
			jc.syncToEdge(watch.Deleted, obj)
		},
	})

	jc.serviceLister = serviceInformer.Lister()
	jc.serviceStoreSynced = serviceInformer.Informer().HasSynced

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    jc.addPod,
		UpdateFunc: jc.updatePod,
		DeleteFunc: jc.deletePod,
	})

	jc.podStore = podInformer.Lister()
	jc.podStoreSynced = podInformer.Informer().HasSynced

	return jc, nil
}
