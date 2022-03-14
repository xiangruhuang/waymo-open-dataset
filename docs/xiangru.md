# My trace of installation on python 3.7.

Don't use `tf 2.4.0` since keras does not have a version that matches.

## Step 1: install tensorflow & waymo-od (v1.2)
```
pip3 install waymo-open-dataset-tf-2-5-0 --user
```

## Step 2: install bazel (sudo)
```
sudo apt-get install --assume-yes pkg-config zip g++ zlib1g-dev unzip python3 python3-pip
BAZEL_VERSION=3.1.0
wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
sudo apt install build-essential
```

### user level installation:
```
BAZEL_VERSION=3.1.0
wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh
bash bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh --user
```
This process will tell you where bazel is located. Make sure PATH contains that folder.

```
./configure.sh
```
with this error ignored: `update-alternatives: error: error creating symbolic link '/etc/alternatives/python3.dpkg-tmp': Permission denied`.

## Step 3: build wheel for waymo-od v1.3
```
export PYTHON_VERSION=3
export PYTHON_MINOR_VERSION=7
export PIP_MANYLINUX2010=0
export TF_VERSION=2.5.0
./pip_pkg_scripts/build.sh
./pip_pkg_scripts/build_pip_wheels.sh ./wheels/ 3
```

## Step 4: install the wheels
```
pip install wheels/waymo_open_dataset_tf_2_5_0-1.4.3-cp37-cp37m-linux_x86_64.whl
```
