"""
Flexiv robot ethernet communication python wrapper
 
Author: Junfeng Ding, Wenxin Du
"""
import copy
import os
import sys
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
# import torch
from scipy.spatial.transform import Rotation as R

from . import libwrapper
from .libwrapper import FvrEthernetConnectionWrapper as fvr_ether
from .libwrapper import FvrParamsWrapper as fvr_params
from .libwrapper import FvrUtilsWrapper as fvr_utils
from .libwrapper.FvrEthernetConnectionWrapper import ExtCtrlMode as ext_ctrl_mode

from .pose import compute_angle_two_quat


class ForceServer(threading.Thread):
    """
    Force Seg Record Class
    
    This class spawns a thread for recording fixed length of force torques.
    """
    def __init__(self, lock, robot, force_seg, length):
        threading.Thread.__init__(self)
        self.force_seg = force_seg
        self.lock = lock
        self.length = length
        self.robot = robot
        self.count = 0

    def run(self):
        while True:
            robot_status_data = fvr_ether.RobotStatusData(
            )
            self.robot.readRobotStatus(robot_status_data)
            force = robot_status_data.m_tcpWrench[:6]
            self.lock.acquire()
            if len(self.force_seg) < self.length:
                force.append(time.time())
                self.force_seg.append(force)
            else:
                self.force_seg.pop(0)
                force.append(time.time())
                self.force_seg.append(force)
            self.count += 1
            time.sleep(0.001)
            self.lock.release()


class ForceClient(threading.Thread):
    """
    Force Seg Fetch Class
    
    This class spawns a thread for fetching fixed length of force torques.
    """
    def __init__(self, lock, force_seg, length):
        threading.Thread.__init__(self)
        self.force_seg = force_seg
        self.lock = lock
        self.length = length

    def run(self):
        while True:
            self.lock.acquire()
            self.lock.release()


class FlexivRobot():
    """
    Flexiv Robot Control Class
    
    This class provides python wrapper for Flexiv robot control in different modes.
    Features include:
        - torque control mode
        - online move mode
        - plan execution mode
        - get robot status information
        - force torque record
    """
    def __init__(self, robot_ip_address, pc_ip_address):
        """ initialize
    
        Args:
            robot_ip_address: robot_ip address string
            pc_ip_address: pc_ip address string
        
        Returns:
            None
    
        Raises:
            RuntimeError: error occurred when ip_address is None.
        """
        self.robot = fvr_ether.RobotClient()
        self.robot.init(robot_ip_address, pc_ip_address, 7)
        time.sleep(0.6)
        
        self.str_mode_dict = {
            'cartesian':  ext_ctrl_mode.CTRL_MODE_CARTESIAN_POSE, # cartesian, torque
            'online': ext_ctrl_mode.CTRL_MODE_ONLINE_MOVE, 
            'idle': ext_ctrl_mode.CTRL_MODE_IDLE,
            'plan': ext_ctrl_mode.CTRL_MODE_PLAN_EXECUTION,
            'line': ext_ctrl_mode.CTRL_MODE_MOVE_LINE,
            'pvat_stream': ext_ctrl_mode.CTRL_MODE_JOINT_PVAT_STREAMING,
            'pvat_download': ext_ctrl_mode.CTRL_MODE_JOINT_PVAT_DOWNLOAD,
            'torque': ext_ctrl_mode.CTRL_MODE_JOINT_TORQUE
        }
        self.mode_str_dict = {v: k for k, v in self.str_mode_dict.items()}
        print('init success')

    @property
    def ctrl_mode(self):
        """
        get the current control mode
        """
        return self.robot.getCtrlMode()
    
    def mode_to_str(self, mode):
        """
        convert mode enum items to strings
        """
        if mode not in self.mode_str_dict:
            return None 
        return self.mode_str_dict[mode]
    
    def str_to_mode(self, s):
        """
        convert mode strings to enum items
        """
        if s not in self.str_mode_dict:
            return None 
        return self.str_mode_dict[s]
    
    @property
    def robot_status(self):
        robot_status_data = fvr_ether.RobotStatusData(
        )
        self.robot.readRobotStatus(robot_status_data)
        return robot_status_data

    @property
    def system_status(self):
        system_status_data = fvr_ether.SystemStatusData(
        )
        self.robot.readSystemStatus(system_status_data)
        return system_status_data

    def switch_mode(self, mode, sleep_time=0.001, max_times=1000, log_interv=100):
        """switch to different control modes
    
        Args:
            mode: 'torque', 'online', 'plan', 'idle', 'line'
            sleep_time: sleep time to control mode switch time
        
        Returns:
            None
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        print("trying to enter the mode {}...".format(mode))
        if mode not in self.str_mode_dict:
            print("mode {} not found, mode switching cancelled.".format(mode))
            return False
        
        control_mode = self.str_to_mode(mode)
        self.robot.setControlMode(control_mode)
        tried_times = 0

        if self.ctrl_mode == control_mode:
            print("already in the mode {}, finished.".format(self.ctrl_mode))
            return True
        if self.ctrl_mode != self.str_to_mode('idle') and mode != 'idle':
            print("not in the mode idle, now swtiching the mode to idle first...")
            self.switch_mode('idle')
        
        self.robot.setControlMode(control_mode)
        tried_times = 0        
        while self.ctrl_mode != control_mode:
            if tried_times > max_times:
                print("max times exceeded, mode switching failed.")
                return False
            if tried_times % log_interv == 0:
                print("switching the mode, current mode: {}".format(self.ctrl_mode))
            tried_times += 1
            time.sleep(sleep_time)
        
        print('Into Mode: {}'.format(str(self.ctrl_mode)))
        return True
    
    def write_io(self, port, value):
        assert port >= 0 and port <= 15
        self.robot.writeDigitalOutput(port, value)

    @property
    def emergency_state(self):
        """get robot is emergency stopped or not
        
        Returns:
            True indicates robot is not stopped, False indicates robot is emergency stopped. 
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return not self.system_status.m_emergencyStop

    @property
    def tcp_pose(self):
        """get current robot's tool pose in world frame

        Returns:
            7-dim list consisting of (x,y,z,rw,rx,ry,rz)
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self.robot_status.m_tcpPose)

    @property
    def tcp_force(self):
        """get current robot's tool force torque wrench

        Returns:
            6-dim list consisting of (fx,fy,fz,wx,wy,wz)
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self.robot_status.m_tcpWrench[:6])

    @property 
    def camera_pose(self):
        """get current wrist camera pose in world frame
    
        Returns:
            7-dim list consisting of (x,y,z,rw,rx,ry,rz)
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self.robot_status.m_camPose)

    @property
    def joint_pos(self):
        """get current joint value
    
        Returns:
            7-dim numpy array of 7 joint position
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self.robot_status.m_jntPos)

    @property
    def joint_vel(self):
        """get current joint velocity
    
        Returns:
            7-dim numpy array of 7 joint velocity
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self.robot_status.m_jntVel)

    @property
    def plan_info(self):
        """get current robot's running plan info
        
        Returns:
            name string of running node in plan 
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        planInfo = fvr_ether.PlanInfoData()
        self.robot.getPlanInfo(planInfo)
        return str(planInfo.m_ptName)

    def is_reached(self, target, trans_epsilon=0.0002, rot_epsilon=0.5):
        """check if the tcp is at the given target position
    
        Args:
            target: 7-dim list or numpy array of 7D target pose (x,y,z,rw,rx,ry,rz)
            trans_epsilon: unit: meter, translation threshold to judge whether reach the target x,y,z
            rot_epsilon: unit: degree, rotation threshold to judge whether reach the target rotation degree
        
        Returns:
            True indicates reaching target pose, False indicates not.
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        target = np.array(target)
        current_pose = self.tcp_pose
        if compute_angle_two_quat(
                current_pose[3:], target[3:]) < rot_epsilon and sum(
                    abs(current_pose[:3] - target[:3]) > trans_epsilon) == 0:
            return True
        else:
            return False

    def is_moving(self, epsilon=1e-2):
        """check whether robot is moving or not 

        Args:
            epsilon: unit: degree/s, threshold to judge whether robot is moving

        Returns:
            True indicates robot is moving, False indicates robot is not running.
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        vel = abs(self.joint_vel)
        if sum(vel > epsilon) == 0:
            return False
        else:
            return True

    def stream_joint_PVAT(self, pos, vel, acc=np.zeros(7), ts=1):
        self.robot.streamJointPVAT(pos, vel, acc, ts)
    
    def send_joint_PVAT(self, pos, vel, acc, ts):
        self.robot.sendJointPVAT(pos, vel, acc, ts)
        
    def send_tcp_pose(self,
                      tcp,
                      wrench=np.array([25.0, 25.0, 10.0, 20.0, 20.0, 20.0]),
                      time_count=0.0,
                      index=1):
        """ make robot move towards target pose in torque control mode, combining with sleep time makes robot move smmothly.
    
        Args:
            tcp: 7-dim list or numpy array, target pose (x,y,z,rw,rx,ry,rz) in world frame
            wrench: 6-dim list or numpy array, max moving force (fx,fy,fz,wx,wy,wz)
        
        Returns:
            None
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        self.robot.sendTcpPose(tcp, wrench, time_count, index)

    def get_tcp_pose(self):
        return self.tcp_pose

    # TODO: move into examples
    def move_cartesian(self,
                    tcp,
                    wrench=np.array([25.0, 25.0, 10.0, 20.0, 20.0, 20.0]),
                    trans_epsilon=0.001,
                    rot_epsilon=0.5):
        """ move in torque control mode until reaching the target.
    
        Args:
            tcp: 7-dim list or numpy array, target pose (x,y,z,rw,rx,ry,rz) in world frame
            wrench: 6-dim list or numpy array, max moving force (fx,fy,fz,,wx,wy,wz)
            trans_epsilon: unit: meter, translation threshold to judge whether reach the target x,y,z
            rot_epsilon: unit: degree, rotation threshold to judge whether reach the target rotation degree
        
        Returns:
            None
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        # self.switch_mode('cartesian')
        time_count = 0.0
        index = 1
        if not self.switch_mode('cartesian'):
            print("mode switching failed, move_cartesian stopped.")
        while not self.is_reached(tcp, trans_epsilon, rot_epsilon):
            self.send_tcp_pose(tcp, wrench, time_count, index)
            time.sleep(0.1)
        print('Reached')

    def send_online_pose(self,
                    tcp,
                    target_twist,
                    max_v=0.05,
                    max_a=0.2,
                    max_w=1.5,
                    max_dw=2.0,
                    index=0):
        """ 
        to be written
        """
        if (self.fl.robot.sendOnlinePose(tcp, target_twist, \
            max_v, max_a, max_w, max_dw, index) != fvr_ether.FvrSt.FVR_OK):
            return False
        return True

    def send_move_line_waypoints(self, waypoints, max_v,\
         max_a, max_w, max_dw, level):
        self.robot.sendMoveLineWaypoints(waypoints, max_v, max_a, max_w, max_dw,
                                level)
    
    # TODO: move into examples
    def move_line(self, waypoints, maxV, maxA, maxW, maxdW, level):
        """ follow given trajectory composed of waypoints in online mode until finishing. 
    
        Args:
            waypoints: n*7 numpy array composed of n waypoints, which is target pose (x,y,z,rw,rx,ry,rz) in world frame
            max_v: n-dim list of double, max linear velocity
            max_a: n-dim list of double, max linear acceleration
            max_w: n-dim list of double, max angular velocity
            max_dw: n-dim list of double, max angular acceleration
            level: n-dim list of int, control level

        Returns:
            None
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        # print("changing mode...")
        if self.robot.getCtrlMode() != ext_ctrl_mode.CTRL_MODE_MOVE_LINE:
            # print("current mode: ", self.robot.getCtrlMode())
            self.switch_mode('line')
        print("Sending way points...")
        self.robot.sendMoveLineWaypoints(waypoints, maxV, maxA, maxW, maxdW,
                                         level)
        while not self.robot.targetReached():
            # print("not reached yet...")
            time.sleep(0.01)
        print('Reached')

    # TODO: move into tool_control.py
    def move_tool_rotation(self, trans):
        """ compute transformed pose after rotation in tool frame
    
        Args:
            trans : 3-dim list or numpy array, (drx, dry, drz) relative to current tool frame
        
        Returns:
            7-dim list consisting of (x,y,z,rw,rx,ry,rz), transformed pose in world frame 
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        tool_pose = self.tcp_pose
        r1 = R.from_quat(
            [tool_pose[4], tool_pose[5], tool_pose[6], tool_pose[3]])
        r2 = R.from_rotvec(np.array(trans))
        r = r1 * r2
        r = r.as_quat()

        tool_pose[3] = r[3]
        tool_pose[4] = r[0]
        tool_pose[5] = r[1]
        tool_pose[6] = r[2]

        return tool_pose

    # TODO: move into tool_control.py
    def move_tool_trans(self, trans):
        """ compute transformed pose after translation in tool frame
    
        Args:
            trans : 3-dim list or numpy array, (dx, dy, dz) relative to current tool frame
        
        Returns:
            7-dim list consisting of (x,y,z,rw,rx,ry,rz), transformed pose in world frame 
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        tool_pose = self.tcp_pose
        r = R.from_quat(
            [tool_pose[4], tool_pose[5], tool_pose[6], tool_pose[3]])
        delta = r.apply(trans)
        return np.array([
            tool_pose[0] + delta[0], tool_pose[1] + delta[1],
            tool_pose[2] + delta[2], tool_pose[3], tool_pose[4], tool_pose[5],
            tool_pose[6]
        ])

    # TODO: move into tool_control.py
    def move_relative_trans(self, pose, trans):
        """ compute translated pose relative to given pose in given pose's frame
    
        Args:
            pose: 7-dim list consisting of (x,y,z,rw,rx,ry,rz), given pose in world frame 
            trans : 3-dim list or numpy array, (dx, dy, dz) relative to current tool frame
        
        Returns:
            7-dim list consisting of (x,y,z,rw,rx,ry,rz), transformed pose in world frame 
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        tool_pose = pose
        r = R.from_quat(
            [tool_pose[4], tool_pose[5], tool_pose[6], tool_pose[3]])
        delta = r.apply(trans)
        return np.array([
            tool_pose[0] + delta[0], tool_pose[1] + delta[1],
            tool_pose[2] + delta[2], tool_pose[3], tool_pose[4], tool_pose[5],
            tool_pose[6]
        ])
    
    # TODO: move into tool_control.py
    def move_relative_rotation(self, pose, rot):
        """ compute translated pose relative to given pose in given pose's frame
    
        Args:
            pose: 7-dim list consisting of (x,y,z,rw,rx,ry,rz), given pose in world frame 
            trans : 3-dim list or numpy array, (dx, dy, dz) relative to current tool frame
        
        Returns:
            7-dim list consisting of (x,y,z,rw,rx,ry,rz), transformed pose in world frame 
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        tool_pose = pose
        r1 = R.from_quat(
            [tool_pose[4], tool_pose[5], tool_pose[6], tool_pose[3]])
        r2 = R.from_rotvec(np.array(rot))
        r = r1 * r2
        r = r.as_quat()

        tool_pose[3] = r[3]
        tool_pose[4] = r[0]
        tool_pose[5] = r[1]
        tool_pose[6] = r[2]

        return tool_pose
    
    def move_joint(self, target_joint_pos, duration, interv=0.01):
        self.switch_mode('pvat_stream')
        targ_pos = target_joint_pos
        cur_pos = self.joint_pos
        v = (targ_pos - cur_pos)/duration
        t_last = time.time()
        t = 0
        while (t < duration):
            t_cur = time.time()
            t_segm = t_cur - t_last
            t_last = t_cur 
            t += t_segm

            cur_pos += v * interv
            # print("cur:{}, targ:{}, vel:{}".format(cur_pos, targ_pos, v))
            self.stream_joint_PVAT(cur_pos, (v))
            time.sleep(interv)

    # TODO: move into examples
    def force_torque(self, down_fz, target_fz, down_z, v=0.002, mid_f=-3):
        """ get 6-dim force torque with current pose
    
        Args:
            down_fz:  move down z force to accelerate
            target_z: return z force
            down_z: target z position
        
        Returns:
            dict,   data['forces'] length*6 numpy array, force seg
                    data['force'] 6-dim list, force torque
                    data['pose'] 7-dim list, end pose
                    data['target_pose'] 7-dim list, initial pose
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """

        initial = self.tcp_pose
        # for _ in range(3):
        # quickly move down to accelerate
        steps = 500
        forces_train = []
        temp = copy.deepcopy(initial)
        for step in range(steps):
            temp[2] = (down_z - initial[2]) / (steps -
                                                1) * step + initial[2]
            self.send_tcp_pose(temp)
            time.sleep(0.01)
            force = self.tcp_force
            forces_train.append(force)
            if force[2] < down_fz:
                break
        forces_train = np.array(forces_train)
        # judge if inserted
        if self.tcp_pose[2] < down_z:
            data = {}
            data['forces'] = []
            data['force'] = self.tcp_force
            data['pose'] = self.tcp_pose
            data['target_pose'] = initial
            return data
        if len(np.where(forces_train[:, 2] < target_fz)[0]) > 0:
            delta_min = 10
            for f in forces_train:
                if abs(f[2] - target_fz) < delta_min:
                    delta_min = abs(f[2]-target_fz)
                    f_record = f
            data = {}
            data['forces'] = forces_train
            data['force'] = f_record
            data['pose'] = self.tcp_pose
            data['target_pose'] = initial
            return data

        data = {}
        data['forces'] = []
        data['force'] = self.tcp_force
        data['pose'] = self.tcp_pose
        data['target_pose'] = initial
        return data

    def execute_plan_by_name(self, name):
        """ execute plan by name, make sure control mode is switched into 'plan'
    
        Args:
            name: string of plan name
        
        Returns:
            None
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        fvr_utils.checkError(
            self.robot.executePlanByName(name))
        system_status = self.system_status
        # wait until execution begin
        while (system_status.m_programRunning == False):
            self.robot.readSystemStatus(system_status)
            time.sleep(1)
        print('Plan started')
        # wait until execution finished
        while (system_status.m_programRunning == True):
            self.robot.readSystemStatus(system_status)
            plan_name = self.plan_info
            # fvrprint.info('Node: {}'.format(str(plan_name)))
            
            if plan_name == 'endnode':
                self.robot.stopPlanExecution()
                break
            # insert detection
            # if plan_name == 'endnode' and self.insertDetection() == 0:
            #     self.robot.stopPlanExecution()
            #     break
            
            time.sleep(0.01)
        print('Plan stopped')

    def execute_plan_by_index(self, index):
        """ execute plan by index, make sure control mode is switched into 'plan'
    
        Args:
            index: int, index of plan
        
        Returns:
            None
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        fvr_utils.checkError(
            self.robot.setControlMode(
                ext_ctrl_mode.CTRL_MODE_PLAN_EXECUTION))
        fvr_utils.checkError(
            self.robot.executePlanByIndex(index))
        system_status = self.system_status
        # wait until execution begin
        while (system_status.m_programRunning == False):
            self.robot.readSystemStatus(system_status)
            time.sleep(1)
        print('Plan started')
        # wait until execution finished
        while (system_status.m_programRunning == True):
            self.robot.readSystemStatus(system_status)
            time.sleep(1)
        print('Plan stopped')

    def begin_plot(self):
        """ turn on plot window
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        plt.ion()
        plt.clf()

    def plot_force_seg(self, force_seg):
        """ visualize 6-axis force seg in six windows
    
        Args:
            force_seg: 
        
        Returns:
            None
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        plt.clf()
        fx = [f[0] for f in force_seg]
        fy = [f[1] for f in force_seg]
        fz = [f[2] for f in force_seg]
        frx = [f[3] for f in force_seg]
        fry = [f[4] for f in force_seg]
        frz = [f[5] for f in force_seg]
        x = np.arange(len(force_seg))
        ax0 = plt.subplot(2, 3, 1)
        ax1 = plt.subplot(2, 3, 2)
        ax2 = plt.subplot(2, 3, 3)
        ax3 = plt.subplot(2, 3, 4)
        ax4 = plt.subplot(2, 3, 5)
        ax5 = plt.subplot(2, 3, 6)
        ## set axis range
        # ax0.set_ylim([-4, 4])
        # ax1.set_ylim([-2, 2])
        # ax2.set_ylim([-30, 30])
        # ax3.set_ylim([-2, 2])
        # ax4.set_ylim([-2, 2])
        # ax5.set_ylim([-2, 2])
        ax0.plot(x, fx)
        ax1.plot(x, fy)
        ax2.plot(x, fz)
        ax3.plot(x, frx)
        ax4.plot(x, fry)
        ax5.plot(x, frz)
        plt.pause(0.4)

    def end_plot(self):
        """ close plot window

        Returns:
            None
    
        Raises:
            RuntimeError: error occurred when mode is None.
        """
        plt.ioff()
        plt.show()
