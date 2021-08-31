# Transparent-Grasp Data Collector

This is the official data collector for transparent-grasp dataset. The collector is mainly developed by [Tony Fang](https://github.com/Galaxies99) based on the initial project by [Minghao Gou](https://github.com/GouMinghao).

- [Transparent-Grasp Data Collector](#transparent-grasp-data-collector)
  - [Preparation](#preparation)
    - [Hardware Preparation](#hardware-preparation)
    - [Python Requirements](#python-requirements)
    - [Install MeshLab](#install-meshlab)
    - [Install Custom Modules (Optional)](#install-custom-modules-optional)
  - [Data Annotation](#data-annotation)
    - [Object Collection & Fixer Attachment](#object-collection--fixer-attachment)
    - [Data Collection Environment Setup](#data-collection-environment-setup)
    - [Register Objects in PST Tracker](#register-objects-in-pst-tracker)
    - [Scan Objects & Preprocessing 3D Models](#scan-objects--preprocessing-3d-models)
    - [PST Tracking Setting](#pst-tracking-setting)
    - [Full Calibration](#full-calibration)
    - [Camera Calibration (Optional)](#camera-calibration-optional)
    - [Annotation](#annotation)
    - [Real-time Evaluation](#real-time-evaluation)
  - [Data Collection](#data-collection)
    - [Manual Data Collection](#manual-data-collection)
    - [Robot Calibration](#robot-calibration)
    - [Robot Calibration (Custom)](#robot-calibration-custom)
    - [Robot Data Collection](#robot-data-collection)
    - [Robot Data Collection (Custom)](#robot-data-collection-custom)
    - [Object Pose Correction](#object-pose-correction)
    - [Visualization](#visualization)
    - [Metadata Annotation](#metadata-annotation)
    - [Ground-truth Depth Rendering](#ground-truth-depth-rendering)
    - [Surface Normal Generation](#surface-normal-generation)
    - [Postprocessing](#postprocessing)
    - [Wash Data](#wash-data)
  - [Citation](#citation)
  - [Maintenance](#maintenance)

## Preparation

### Hardware Preparation

**Prequisities**: None.

- PST Tracker;
- 3D Scanner;
- RealSense D435;
- RealSense L515;
- Flexiv Arm (or any type of robot arm).

### Python Requirements

**Prequisities**: None.

We recommend you to setup the collector in a `conda` environment.

```bash
conda create --name tgcollector python=3.7
conda activate tgcollector
```

Then, install all the python requirements using the following command.

```bash
pip install -r requirements.txt
```

### Install MeshLab

**Prequisities**: None.

Install [MeshLab Release 2020.06](https://github.com/cnr-isti-vclab/meshlab/releases/tag/Meshlab-2020.06) by compiling the source code.

### Install Custom Modules (Optional)

**Prequisities**: None.

Another library that is required is the `netrelay` from this [repository](https://github.com/galaxies99/netrelay), but you don't need to worry about that, since this repository already contains the important module of the `netrelay` library.

## Data Annotation

### Object Collection & Fixer Attachment

**Prequisities**: None.

Object collection and fixer attachment processes are finished manually. You should give each object a different name to distinguish them.

### Data Collection Environment Setup

**Prequisities**: [Preparation](#hardware-preparation).

Setup your own data collection environment. You can refer to the environment that is described in the paper.

### Register Objects in PST Tracker

**Prequisities**: [Preparation](#hardware-preparation), [Object Collection & Fixer Attachment](#object-collection--fixer-attachment).

Put markers on the fixers, then register each object as its name in the PST tracker .

### Scan Objects & Preprocessing 3D Models

**Prequisities**: [Object Collection & Fixer Attachment](#object-collection--fixer-attachment).

1. Scan the objects using a scanner which can produce `.obj` scanning results;
2. Put the results in `preprocessing/rawdata`, the folder should contain many folders containing an `.obj` file and some optional `.png` files (if color scanning mode is used);
3. List the expected model filename in `object_file_name_list.txt`, one filename per line. Notice that the object file `preprocessing/rawdata/[filename]/[filename].obj` must exist for all `filename.[extension]` in the filename list. 
4. List the models that are scanned in color scanning mode (not black-white mode) in `preprocessing/use_t2v_list.txt` using the same format.
5. Then run the following commands to preprocess the raw data.

   ```bash
   python preprocessing/preprocessing.py --server [Your MeshLab Server Path]
   ```

   Here, replace `[Your MeshLab Server Path]` with your own MeshLab Server path, usually in `[Your Meshlab Path]/distrib/meshlabserver` once you have successfully compiled MeshLab.

### PST Tracking Setting

**Prequisities**: [Register Objects in PST Tracker](#register-objects-in-pst-tracker)

1. Connect the PST tracker with a computer in Windows system, and run PST REST Server according to the documentation of PST tracker. Then run the following command, and keep the running program in background.

    ```bash
    python -m netrelay.stream
    ```

2. Run the followng command to start the relay server.

    ```bash
    python -m netrelay.relay_stream -s [IP Address]:[Port]
    ```

    Notice that:

    - The first command should output a log `{"message": "Server Started"}` in the screen, otherwise you may need to check the connection between the PST tracker and the computer.
    - Replace the `[IP Address]` and the `[Port]` in the second command with the specific IP address and port of the computer (you may need to execute `ipconfig` to get the IP address).

**Notice**. You should keep the PST server running until you do not need the PST tracking service anymore.

### Full Calibration

**Prequisities**: [Data Collection Environment Setup](#data-collection-environment-setup), [PST Tracking Setting](#pst-tracking-setting)

1. Preparing an 7x7 aruco image (id: 0). Register it as "calibration" in the PST tracker and align the axis as you wish.

2. Put the aruco image in the place that PST tracker and camera(s) can track/capture it.

3. Run the following command to perform calibration between camera and tracker.

    ```bash
    python calibration/calibration.py
    ```

    Combining with the data reading from tracker, the calibration process can calculate the transformation matrix between tracker and camera, which will be stored in file `configs/T_tracker_camera.npy`.

    The script will first detect the pose data of aruco picture from camera, then fetch the tracker data from the Windows computer, and then display it on the screen. You can type `y` if you are satisfied with the tracking data, and go to the annotation stage; or `n` if you are not satisfied with the data, and the script will fetch tracker data again and repeat the previous process.

### Camera Calibration (Optional)

**Prequisities**: [Data Collection Environment Setup](#data-collection-environment-setup).

If you just want to perform calibration between cameras, then you can use the aruco image and the `camera_calibration` script in the `calibration` folder.

### Annotation

**Prequisities**: [Scan Object & Preprocessing 3D models](#scan-objects--preprocessing-3d-models), [Full Calibration](#full-calibration).

Execute the following command and begin annotation.

```bash
python annotation/script.py --id [Object ID] --time [Times of sampling] --ip [IP Address] --port [Port]
```

Here, replace the `[Object ID]` with the current object ID (0-based, the same order as in the previous file `object_file_name_list.txt`), replace `[Times of samping]` with the current times of sampling (0-based), replace `[IP Address]` with the same IP address in Step 1, and replace `[Port]` with the same port in Step 1. Here is the execution process of the script.

- The script will first fetch the tracker data from the Windows computer, and then display it on the screen. You can type `y` if you are satisfied with the tracking data, and go to the annotation stage; or `n` if you are not satisfied with the data, and the script will fetch tracker data again and repeat the previous process.
- After you enter in the annotation stage, a GUI window will be displayed on the screen. You can annotate the data according to the guide on the top-left of the window.
- When `time` is `0`, you may perform the annotation process from beginning. Otherwise, the program will calculate the pose according to the pose reading from tracker, and then you just need to fine-tune it.

**Notice**. The camera will capture real-time images, so in order to get a correct 6dpose, make sure that the camera won't move during annotation.

### Real-time Evaluation

**Prequisities**: [Annotation](#annotation).

Run the following script to evaluate your annotation in real-time.

```bash
python annotation/eval_realtime.py --id [Object ID] --ip [IP Address] --port [Port]
```

Here, replace the `[Object ID]` with the current object ID (0-based, the same order as in the previous file `object_file_name_list.txt`), replace `[IP Address]` with the same IP address in Step 1, and replace `[Port]` with the same port in Step 1. After several seconds, you will see real-time evluation image captured by the tracker on the screen.

Notice that you can also ignore the `--id [Object ID]` arguments. By doing so, the evaluation process will detect the objects automatically and load the corresponding models.

## Data Collection

### Manual Data Collection

**Prequisities**. [Data Annotation](#data-annotation).

After annotating all the objects you need for constructing data, run the following script to collects the data manually.

```bash
python collection/data_collector.py --id [Scene ID]
```

Once you are satisfied with the picture and the models shown on the screen, you can press Enter to save one shot of this screen. You can also press `,` or `.` to increase or decrease the transparency of the objects respectively. By pressing `q`, you can finish the collection of this scene. The data will be collected in the following form.

```
data
├── scene1
|   ├── 0
|   |   ├── rgb1.png
|   |   ├── rgb2.png
|   |   ├── depth1.png
|   |   ├── depth2.png
|   |   └── pose
|   |       ├── 0.npy
|   |       ├── 23.npy
|   |       └── ...
|   └── ...
├── scene2
|   └── ...
└── ...
```

- `rgb1.png` and `depth1.png` are the RGB image and the depth image read from Realsense D435;
- `rgb2.png` and `depth2.png` are the RGB image and the depth image read from Realsense L515;
- `pose` contains are the objects detected in `image1.png`; `[ID].npy` denotes the pose of object numbered `[ID]` in the picture.

### Robot Calibration

**Prequisities**. [Preparation](#preparation)

The default settings for robot calibration is using Flexiv Arm. You should make some necessary settings to the Flexiv Arm such as workspace, workload, etc.

Run the following script:

```bash
python flexiv_robot/trace-calibration.py
```

Then, you just need to wait for the calibration process to finish. The calibration data is stored in the `configs/robot_calibration` folder.

**Custom Path**. For custom robot path, edit `robot/robot_path/joint_poses.npy` to specify the joint settings of each waypoint.

### Robot Calibration (Custom)

**Prequisities**. [Preparation](#preparation)

To automatically collect data by robots, you need to first call the robot calibration script in every point of the robot's route by

```bash
python calibration/robot_calibration.py --id [ID] --path [Path to Robot Image]
```

where `[ID]` is the route point's ID, and `[Path to Robot Image]` is the image path to store the calibration image (default: `robot_images`). After executing the script, the program will generate the calibration image in the image folder, while saving the calibration transformation matrix (from camera to calibration object) in `configs/robot_calibration/` folder. This calibration process uses `aruco` to perform camera calibration.

### Robot Data Collection

**Prequisities**. [Preparation](#preparation), [Data Annotation](#data-annotation), [Robot Calibration](#robot-calibration).

The default settings for robot data collection is using Flexiv Arm. You should make some necessary settings to the Flexiv Arm such as workspace, workload, etc.

Run the following script:

```bash
python collection/trace.py
```

Then, after entering scene ID, you just need to wait for the data collection process to finish. The data will be stored in `data/scene[Scene ID]` folder.

### Robot Data Collection (Custom)

**Prequisities**. [Preparation](#preparation), [Data Annotation](#data-annotation), [Robot Calibration (Custom)](#robot-calibration-custom).

If you want to automatically collect data by robots, you can follow the annotation step 1-2, then call the data collector script in every point of the robot's route by

```bash
python collection/robot_collector.py --id [Scene ID] --time [Times of sampling] (--ip [IP Address]) (--port [Port])
```

Here, replace the `[Object ID]` with the current object ID (0-based, the same order as in the previous file `object_file_name_list.txt`), replace `[Times of samping]` with the current times of sampling (0-based), replace `[IP Address]` with the same IP address in Step 1, and replace `[Port]` with the same port in annotation step 1.

The data will be in the same format as introduced in data collection section.

### Object Pose Correction

**Prequisities**. [Robot Data Collection](#robot-data-collection) or [Robot Data Collection (Custom)](#robot-data-collection-custom).

After automatically collecting data, you may need to correct the poses of the object due to the vision field of the tracker. We can use the robot calibration data and the collected data to perform pose correction. You may execute the following script:

```bash
python postprocessing/pose_corrector.py --data_dir [Data Path] --id [Scene ID] --perspective_num [Perspective Number] (--weight_path [Weight Path])
```

The corrected pose will be in the folder named `corrected_pose` in `[Data Path]` directory; `weight_path` is used for pose correction, default to None.

### Visualization

**Prequisities**. [Robot Data Collection](#robot-data-collection) or [Robot Data Collection (Custom)](#robot-data-collection-custom).

You may perform visualization to check whether the collected data is satisfactory by

```bash
python collection/visualization.py --data_dir [Scene Path] --id [Image ID] (--corrected) (--weight_path [Weight Path])
```

where setting `--corrected` means using the corrected poses (the pose will be automatically corrected, there is no need to perform "object pose correction" first), otherwise the default poses detected by the tracker will be used; `weight_path` is used for pose correction, default to None.

**Notice**. If you perform visualization along with a `--corrected` option, the pose will be automatically corrected. There is no need to perform pose correction first.

### Metadata Annotation

**Prequisities**. [Robot Data Collection](#robot-data-collection) or [Robot Data Collection (Custom)](#robot-data-collection-custom).

After collection, you need to manually label metadata for each scenes, since some poses generated by the tracker or the pose corrector may be inaccurate. Execute the following script:

```bash
python postprocessing/metadata_annotator.py --data_dir [Data Path] --id [Scene ID] --camera_calibration_file [Camera Calibration File] (--corrected) (--weight_path [Weight Path])
```

where setting `--corrected` means using the corrected poses (the pose will be automatically corrected, there is no need to perform "object pose correction" first), otherwise the default poses detected by the tracker will be used; `weight_path` is used for pose correction, default to None.

**Notice**. If you perform metadata annotation along with a `--corrected` option, the pose will be automatically corrected and saved in `corrected_pose` folder in each image folder.

### Ground-truth Depth Rendering

**Prequisities**. [Metadata Annotation](#metadata-annotation).

You may render the ground-truth depth by executing the following command:

```bash
python postprocessing/depth_renderer.py --image_path [Image Path] (--corrected) (--weight_path [Weight Path])
```

where setting `--corrected` means using the corrected poses (the pose will be automatically corrected, there is no need to perform "object pose correction" first), otherwise the default poses detected by the tracker will be used; `weight_path` is used for pose correction, default to None.

### Surface Normal Generation

**Prequisities**. [Ground-truth Depth Rendering](#ground-truth-depth-rendering).

you may generate the surface normal by executing the following command:

```bash
python postprocessing/sn_generator.py --data_dir [Data Path] --begin_id [Begin ID] --end_id [End ID]
```

where `[Begin ID]` and `[End ID]` are the begin scene ID and the end scene ID of the scenes on which we want to perform surface normal generation.

### Postprocessing

**Prequisities**. [Metadata Annotation](#metadata-annotation).

You may perform ground-truth depth rendering and surface normal generation automatically by executing the postprocessing script.

```bash
python postprocessing/postprocessing.py --data_dir [Data Path] --begin_id [Begin ID] --end_id [End ID]
```

where `[Begin ID]` and `[End ID]` are the begin scene ID and the end scene ID of the scenes on which we want to perform postprocessing.

### Wash Data

**Prequisities**. [Postprocessing](#postprocessing).

You may wash invalid data (those data with depth = 0 for all pixels) by executing the data washer script.

```bash
python postprocessing/wash_data.py --data_dir [Data Path] --begin_id [Begin ID] --end_id [End ID]
```

where `[Begin ID]` and `[End ID]` are the begin scene ID and the end scene ID of the scenes on which we want to perform postprocessing.

## Citation

```bibtex
@misc{fang2021tgcollector,
  author =       {Hongjie Fang, Minghao Gou, Sheng Xu, Hao-Shu Fang},
  title =        {Transparent-Grasp Data Collector},
  howpublished = {\url{https://github.com/Galaxies99/transparent-grasp-data-collector}},
  year =         {2021}
}
```

## Maintenance

The repository is currently maintained by [Hongjie Fang](https://github.com/Galaxies99). Raise an issue if you encounter any problem.
