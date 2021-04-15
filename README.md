# abandoned_object_detection

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
4. Add ~/.local/bin directory in your PATH.
5. Comfirm docker-compose installed successfully.
  ```
  $ docker-compose --version
  ```
6. Add "default-runtime": "nvidia" to your /etc/docker/daemon.json configuration file to build the TensorRT plugin. **And restart the docker service or reboot your systme.**
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
  $ chmod +x ./scripts/*.sh
  ```

Usage
-----

Start the services of the application
```
$ ./scripts/compose-up.sh
```

Stop the services of the application
```
$ ./scripts/compose-down.sh
```

Licenses
--------

- This application was inherited the TensorRT based YOLO implementation from [jkjung-avt](https://github.com/jkjung-avt)/[tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos). The software is under the MIT license.
- The [jkjung-avt](https://github.com/jkjung-avt)/[tensorrt_demos](https://github.com/jkjung-avt/tensorrt_demos) code of [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT) samples to develop most of the demos in this repository.  Those NVIDIA samples are under [Apache License 2.0](https://github.com/NVIDIA/TensorRT/blob/master/LICENSE).
