# 6dPose Annotator for Camera

This project is initially developed by [Minghao Gou](https://github.com/GouMinghao) and then modified by [Tony Fang](https://github.com/Galaxies99). This project aims to develop a convenient manual annotator when raw models and scenes are given. In this special camera version of 6dpose annotator, the scene is captured by a RealSense camera.

## Preparation

Install all the python requirements using the following command.

```bash
pip install -r requirements.txt
```

Install [MeshLab Release 2020.06](https://github.com/cnr-isti-vclab/meshlab/releases/tag/Meshlab-2020.06) by compiling the source code.

##  Preprocessing

Put the raw data in `preprocessing/rawdata`, and list the expected model filename in `object_file_name_list.txt`, one filename per line. Then run the following commands to pre-process the raw data.

```
python preprocessing/preprocessing.py --server [Your MeshLab Server Path]
```

Here, replace `[Your MeshLab Server Path]` with your own MeshLab Server path, usually in `[Your Meshlab Path]/distrib/meshlabserver` once you have successfully compiled MeshLab.

## Run

Run the following commands <u>in parallel</u>.

```bash
python camera/[Camera Version].py
```

```
python annotator.py
```

Here, replace `[Camera Version]` with your RGB-D camera version. Currently, `realsense_D435` and `realsense_L515` are supported.

Then, you can annotate the data according to the guide in GUI window.

**Notice**. The camera will capture real-time images, so in order to get a correct 6dpose, make sure that the camera won't move during annotation.