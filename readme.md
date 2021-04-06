# 6dPose Annotator for Camera

This project is initially developed by [Minghao Gou](https://github.com/GouMinghao) and then modified by [Tony Fang](https://github.com/Galaxies99). This project aims to develop a convenient manual annotator when raw models and scenes are given, along with a handy data collector. In this special camera version of 6dpose annotator, the scene is captured by a RealSense camera.

## Preparation

Install all the python requirements using the following command.

```bash
pip install -r requirements.txt
```

Install [MeshLab Release 2020.06](https://github.com/cnr-isti-vclab/meshlab/releases/tag/Meshlab-2020.06) by compiling the source code.

Other than the previous requirements, you also need to install `ur_toolbox` from this [repository](https://github.com/graspnet/ur_toolbox). Another library that is required is the `netrelay` from this [repository](https://github.com/galaxies99/netrelay), but you don't need to worry about that, since this repository already contains the important module of the `netrelay` library.

##  Preprocessing

Scan the object using the scanner, and then put the raw data in `preprocessing/rawdata`, and list the expected model filename in `object_file_name_list.txt`, one filename per line. Notice that the object file `preprocessing/rawdata/[filename]/[filename].obj` must exist for all `filename.[extension]` in the filename list. List the models that are in colored mode (not black-white mode) in `preprocessing/use_t2v_list.txt` using the same format. Then run the following commands to preprocess the raw data.

```
python preprocessing/preprocessing.py --server [Your MeshLab Server Path]
```

Here, replace `[Your MeshLab Server Path]` with your own MeshLab Server path, usually in `[Your Meshlab Path]/distrib/meshlabserver` once you have successfully compiled MeshLab.

## Run

**Step 1**. Connect the PST tracker with a computer in Windows system, and run PST REST Server according to the documentation of PST tracker. Then run the following command, and keep the running program in background. 

```bash
python -m netrelay.stream
```
After that, run the followng command to start the relay server.

```bash
python -m netrelay.relay_stream -s [IP Address]:[Port]
```

Notice that:

- The first command should output a log `{"message": "Server Started"}` in the screen, otherwise you may need to check the connection between the PST tracker and the computer.
- Replace the `[IP Address]` and the `[Port]` in the second command with the specific IP address and port of the computer (you may need to execute `ipconfig` to get the IP address).

**Step 2**. Run the following command, and keep the running program in background. 

```bash
python camera/[Camera Version].py
```

Here, replace `[Camera Version]` with your RGB-D camera version. Currently, `realsense_D435` and `realsense_L515` are supported.

**Step 3**. Run the following command to perform calibration between camera and tracker.

```bash
cd calibration
python calibration.py
```

This calibration process uses `aruco` to perform camera calibration. Combining with the data reading from tracker, the calibration process can calculate the transformation matrix between tracker and camera, which will be stored in file `configs/T_tracker_camera.npy`.

The script will first detect the pose data of aruco picture from camera, then fetch the tracker data from the Windows computer, and then display it on the screen. You can type `y` if you are satisfied with the tracking data, and go to the annotation stage; or `n` if you are not satisfied with the data, and the script will fetch tracker data again and repeat the previous process.

**Notice**. Make sure to use aruco 7x7 database and pictures with id 0.

**Step 4**. Execute the following command and begin annotation.

```bash
python script.py --id [Object ID] --time [Times of sampling] --ip [IP Address] --port [Port]
```

Here, replace the `[Object ID]` with the current object ID (0-based, the same order as in the previous file `object_file_name_list.txt`), replace `[Times of samping]` with the current times of sampling (0-based), replace `[IP Address]` with the same IP address in Step 1, and replace `[Port]` with the same port in Step 1. Here is the execution process of the script.

- The script will first fetch the tracker data from the Windows computer, and then display it on the screen. You can type `y` if you are satisfied with the tracking data, and go to the annotation stage; or `n` if you are not satisfied with the data, and the script will fetch tracker data again and repeat the previous process.
- After you enter in the annotation stage, a GUI window will be displayed on the screen. You can annotate the data according to the guide on the top-left of the window.
- When `time` is `0`, you may perform the annotation process from beginning. Otherwise, the program will calculate the pose according to the pose reading from tracker, and then you just need to fine-tune it.

**Notice**. The camera will capture real-time images, so in order to get a correct 6dpose, make sure that the camera won't move during annotation.

## Annotation Evaluation

Follow the Step 1 to Step 3 in the "Run" section, then run the following script to evaluate your annotation in real-time.

```bash
python eval_realtime.py --id [Object ID] --ip [IP Address] --port [Port]
```

Here, replace the `[Object ID]` with the current object ID (0-based, the same order as in the previous file `object_file_name_list.txt`), replace `[IP Address]` with the same IP address in Step 1, and replace `[Port]` with the same port in Step 1. After several seconds, you will see real-time evluation image captured by the tracker on the screen.

Notice that you can also ignore the `--id [Object ID]` arguments. By doing so, the evaluate process will detect the objects automatically and load the corresponding models.

## Data Collection

After annotating all the objects you need for constructing data, run the following script to collects the data.

```bash
python data_collector.py --id [Scene_ID]
```

This script will read photo stream from two separate cameras: Realsense D435 and Realsense L515. So make sure to run the following commands on background.

```bash
python camera/realsense_D435.py
python camera/realsense_L515.py
```

Once you are satisfied with the picture and the models shown on the screen, you can press Enter to save one shot of this screen. You can also press `,` or `.` to increase or decrease the transparency of the objects respectively. By pressing `q`, you can finish the collection of this scene. The data will be collected in the following form.

```
data
├── scene1
|   ├── 0
|   |   ├── image1.png
|   |   ├── image2.png
|   |   ├── image_depth1.png
|   |   ├── image_depth2.png
|   |   └── pose
|   |       ├── 0.npy
|   |       ├── 23.npy
|   |       └── ...
|   └── ...
├── scene2
|   └── ...
└── ...
```

- `image1.png` and `image_depth1.png` are the RGB image and the depth image read from Realsense D435;
- `image2.png` and `image_depth2.png` are the RGB image and the depth image read from Realsense L515;
- `pose` contains are the objects detected in `image1.png`; `[ID].npy` denotes the pose of object numbered `[ID]` in the picture.

## Maintenance

Mailto: galaxies@sjtu.edu.cn