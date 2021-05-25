source /home/minghao/anaconda3/etc/profile.d/conda.sh
conda activate mink38
cd ~/hongjie/git/6dpose-annotator-camera/calibration
python robot_calibration.py --id $1
