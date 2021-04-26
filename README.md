# abandoned_object_detection

Table of Contents
-----------------
- [Description](#description)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Licenses](#licenses)

Description
-----------
This is an Abandoned Object Detection (AOD) using a TensorRT-optimized YOLOv4(416 frame size) as the object detection model.

Here's an example of the screenshot of the AOD demo.

・First display : object detection

<img src="https://github.com/MACNICA-CLAVIS-NV/abandoned_object_detection/blob/master/pictures/ObjectDetection_backpack_cellphone_person.png">

・Second display : "warning" text and yellow bounding box

<img src="https://github.com/MACNICA-CLAVIS-NV/abandoned_object_detection/blob/master/pictures/warning_backpack_cellphone.png">

・Last display : "abandoned" and red bounding box

<img src="https://github.com/MACNICA-CLAVIS-NV/abandoned_object_detection/blob/master/pictures/abandoned_backpack_cellphone.png">


Prerequisites
-------------

- NVIDIA Jetson with JetPack 4.5 or later
- USB camera

Installation
------------

Install docker-compose (You need to have docker-compose with the runtime paramter support.)
1. If you already have pip installed with apt in your Jetson. Remove it.
  ```
  $ sudo apt remove python3-pip
  ```
2. Install pip from PyPA
  ```
  $ sudo apt update
  $ sudo apt install curl python3-testresources
  $ curl -kL https://bootstrap.pypa.io/get-pip.py | python3
  ```
3. Install docker-compose
  ```
  $ python3 -m pip install --user docker-compose
  ```
4. Add $HOME/.local/bin directory in your PATH.
5. Comfirm docker-compose installed successfully.
  ```
  $ docker-compose --version
  ```
6. Add "default-runtime": "nvidia" to your /etc/docker/daemon.json configuration file to build the TensorRT plugin. **And restart the docker service or reboot your system.**
  ```
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
  ```
7. Clone this repository.
  ```
  $ git clone https://github.com/MACNICA-CLAVIS-NV/abandoned_object_detection
  ```
8. Add the execution permission to the shell scripts.
  ```
  $ cd abandoned_object_detection
  $ chmod +x ./scripts/*.sh
  ```

Usage
-----

Start the services of the application
```
$ ./scripts/compose-up.sh
```
You can see the dashboard at [http://localhost:1880/ui](http://localhost:1880/ui)

Stop the services of the application
```
$ ./scripts/compose-down.sh
```

Troubleshooting
---------------
If you see the following error,
```
inference_1  | -------------------------------------------------------------------
inference_1  | PyCUDA ERROR: The context stack was not empty upon module cleanup.
inference_1  | -------------------------------------------------------------------
inference_1  | A context was still active when the context stack was being
inference_1  | cleaned up. At this point in our execution, CUDA may already
inference_1  | have been deinitialized, so there is no way we can finish
inference_1  | cleanly. The program will be aborted now.
inference_1  | Use Context.pop() to avoid this problem.
inference_1  | -------------------------------------------------------------------
```
please run the PyCUDA auto initialization alone with the following commands.

1. Execute a shell in the conatainer.
```
$ ./scripts/docker_run.sh 
```
2. Run the following python command.
```
# python3 -c "import pycuda.autoinit"
```
3. Exit from the shell in the container.
```
# exit
```

Then restat the abandoned_object_detection application.

Licenses
--------

- This application was inherited the TensorRT based YOLO implementation from [jkjung-avt](https://github.com/jkjung-avt)/[tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos). The software is under the MIT license.
- The [jkjung-avt](https://github.com/jkjung-avt)/[tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos) referenced source code of [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) samples to develop most of the demos in this repository.  Those NVIDIA samples are under [Apache License 2.0](https://github.com/NVIDIA/TensorRT/blob/master/LICENSE).
