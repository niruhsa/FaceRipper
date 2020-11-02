# FaceRipper v1

### What is FaceRipper?
FaceRipper is a tool used to extract faces from large sets of videos to generate face sets for deep learning. It uses euclidean distance to determine how much a face in the frame of a video relates to a face in your reference images. FaceRipper also sets up the folders in a way that you can plug and play with Keras `ImageDataGenerator` and `flow_from_directory` with `categorical` classification.

### Why make this in the first place?
I wanted to make a deep neural network that could let me detect faces using inference since euclidean distance in real time was too slow on high resolution images... Making the datasets manually for this type of task where thousands of images per face were required was a daunting task and wanted to automate the process. I made FaceRipper to take as much manual work out of that process as I could. Of course you will still need to go in and check the final result of each extraction, especially when using high tolerances, but other than that I think it does a pretty good job for me when I'm getting > 50k images per person. In my experience it requires very little intervention if you have enough reference images that are high enough quality per person.


## Usage
To use FaceRipper, python 3 is required (python 2 is untested, and I've known of complications from using DLIB with python 2). There are a few arguments that you can parse in, and some required ones.

Required Arguments:
 * `--config <config file>` The location to the config file that you are using
 * `--video-dir <dir>` The location to the directory of videos to extract faces from
 * `--target-dir <dir>` The location to extract the faces to

Optional Arguments:
 * `--tolerance <0.0-1.0>` The tolerance for the facial detection (See below for explanation). Default is `0.325`.
 * `--threads <thread count>` The number of cores/threads to use for multithreading. Default is `4`.
 * `--jitters <jitter count>` The number of times to jitter face detection on your reference images (See below for explanation). Default is `0`.
 * `--model (cnn/hog)` The face recognition model to use, `hog` is used for CPU based recognition whereas `cnn` is used for GPU based detection using CUDA. `cnn` requires DLIB to be compiled with CUDA support. You cannot use `cnn` with more than 1 thread. Default is `hog`.
 * `--blur <threshold>` The threshold to detect how much blur is acceptable in final extracted faces (See below for explanation). Default is `250`.
 * `--save-blurry` Save blurry images in a .blurry folder inside each label

Example Command: `python src/face_ripper.py --config config.json --video-dir "data/videos" --target-dir "data/faces" --threads 4`

## Explanations
### What's the config file for?
I was tossing up between using input directories as the data source and a config file with more data, ultimately I went with a config file since it lets you add mass people and you can change the label name, you can also generate this config file automatically from a directory full of source images and just append the `label` tag to each item in the config file to change its output path name. The `group` field for now is useless and I've just kept it in there for the future when I might add the ability to change the output directory from `label` to `GROUP___LABEL` per person, so when you do recognition via deep learning the label it recognizes includes the group name that you can easily use string manipulation to get the group and persons name.

### Issues with DLIB
I've experienced some issues with DLIB when testing this, mainly high memory usage when it uses CUDA. To solve this I recommend not building from source, but if you want to, I recommend you follow the instructions in the `INSTALLING_DLIB.md` file. For majority of the people that are going to use this, installing dlib from pip is the easiest way to go, although for windows users I recommend you follow the guide in the `INSTALLING_DLIB.md` file under the Windows section since there are some extra steps you have to follow to even install it from pip.

## Features
### Blur Detection
FaceRipper uses blur detection to get rid of blurry faces that it extracts, this is partly because it's harder for the euclidean distance algorithm to detect the subtle identifying marks of the face when comparing it to the reference images. Of course you can turn this down to 0 if you want to disable it by using `--blur 0` as a command line argument, but I would recommend that you didn't.

### Tolerance
Tolerance is a term that is used to define how closely related a face must be to a reference image in order for it to match. This value starts at `0.0` and can go up to `1.0`, although in rare cases I've seen it go past `1.0`. The default is set to `0.35` which is saying that the face must match within 35% of a reference face for it to be saved. Of course if there is blur on the face that is outside of the blur range it will be excluded, but if the blur is within range it will be saved into the target directory using the label of the reference image.

### Reference Images (Jitter & Model)
For each person that you are matching, I'd recommend atleast 5-10 high quality photos of the person, make sure that you aren't using pictures that are detecting false faces, to test this you can use the `misc/reference_detections.py` file with the same config file to create a GUI popup of each face that it recognizes in the photos. There is a command line argument called `num_jitters` that is used to make the facial detection more stricter, the higher this number the stricter it becomes, but it takes x times longer, e.g `--jitters 100` would be 100x slower than `--jitters 0` which is the default. You can also use the `--jitters <count>` argument in the `misc/reference_detections.py` file to fine tune your reference images. You can also input the `--model` argument to specify to use either CPU or GPU in `misc/reference_detections.py`.

## Datasets
To get the dataset I'm using in `config.json`, head over to [releases](https://github.com/niruhsa/FaceRipper/releases/tag/v0.1) and download the `dataset.zip` file. Place it inside of a folder called `data` in the root directory of this repo.

## For the Future
### TODO
I'd like to add a few things to this project in the future, some of these include:
 * Face rotation correction for tilted faces (This is hard to implement since its hard to detect where the eyes are accurately, especially on smaller images)
 * MTCNN support to rule out lots of false positives, although the use of this with more than 1 thread on a GPU wouldn't work, so it'd be limited to CPU when the thread count is more than 1.
 * Able to extract reference image classifications and save to a file for faster loading times and future re-use instead of having to re-extract on every use (This is the slowest part that I've found, this would be an easy solution, but having multiple faces per label is the hard bit on how to organise the data in the correct way)
 * Clustered machine support. Ability to start a server with the videos to extract from, and run multiple nodes all on other machines to support a fuck tonne of more multicore support and speed up the time it would take to extract a video. This would require port forwarding on the server instance if it is on a public network. It's a nice feature to think about but for now I don't think it's that required.
 * Able to not use a config file and just parse in an input directory of reference images and extract label names from last child directory names.