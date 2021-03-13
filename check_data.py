"""
This file implements the lego reconstruction using incremental method based on marker pose

Author: Haoshu Fang, Xiaoshan Lin
"""
import open3d as o3d
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat
import yaml
import os
import cv2
# from cv2 import aruco
# from scipy.spatial.transform import Rotation as R
# from .averageQuaternions import averageQuaternions
from pyquaternion import Quaternion

MAX_DEPTH_M = 1.3
MIN_DEPTH_M = 0.0
DOWNSAMPLE_VOXEL_SIZE_M = 0.002

FILTER_NUM_NEIGHBOR = 30
FILTER_STD_RATIO = 2.0
FILTER_RADIUS_M = 0.01

ICP_MAX_DISTANCE_COARSE = DOWNSAMPLE_VOXEL_SIZE_M * 10
ICP_MAX_DISTANCE_FINE = DOWNSAMPLE_VOXEL_SIZE_M * 1

NORMAL_RADIUS = 0.01
NORMAL_NUM_NEIGHBOR = 30

PLANE_Z_MIN = -0.005
PLANE_Z_MAX = 0.002
PLANE_Y_MIN = -0.31
PLANE_Y_MAX = 0
PLANE_X_MIN = 0
PLANE_X_MAX = 0.29

MARKER_LENGTH = 54
width = 1280
height = 720
dist_coeffs = np.array([0., 0., 0., 0.]).reshape(4, 1)

INFO = False

def vis(data_path, visualize=True):

    if not os.path.exists(os.path.join(data_path, 'depth_vis')):
        os.mkdir(os.path.join(data_path, 'depth_vis'))
    depth_names = os.listdir(os.path.join(data_path, 'depth'))
    depth_names.sort()
    depth_full_names = [
        os.path.join(data_path, 'depth/' + name) for name in depth_names
    ]
    for i in range(len(depth_full_names)):
        depth_img = cv2.imread(depth_full_names[i],cv2.IMREAD_UNCHANGED).astype(np.float32)
        # cv2.imshow('res', depth_img)
        # cv2.waitKey()
        cv2.imwrite(os.path.join(data_path, 'depth_vis', depth_names[i]),
                    (depth_img*100).astype(np.uint16))
    # assert len(depth_full_names) == len(rgb_full_names)




def lego_icp(cam,data_path,pose_npy_file, visualize=False):
    if cam == 'realsense':
        camera_matrix = np.load(os.path.join(data_path, 'camK.npy'))
    else:
        camera_matrix = np.array([[631.54864502,0.,638.43517329],[0.,631.20751953,366.49904066],[0.,0.,1.]])
    rgb_names = os.listdir(os.path.join(data_path, 'rgb'))
    rgb_names.sort()
    rgb_full_names = [
        os.path.join(data_path, 'rgb/' + name) for name in rgb_names
    ]

    depth_names = os.listdir(os.path.join(data_path, 'depth'))
    depth_names.sort()
    depth_full_names = [
        os.path.join(data_path, 'depth/' + name) for name in depth_names
    ]
    # assert len(depth_full_names) == len(rgb_full_names)

    # poses = get_poses(rgb_full_names, camera_matrix, markers_transformation)
    
    # poses = load_trajectory_npy(os.path.join(data_path, 'end_link_poses.npy'), np.array(7.73315019e-02,5.47893183e-04,6.31583587e-02, 7.02847013e-01, -3.85296346e-04,-7.15370101e-03,7.11304965e-01))
    # poses = load_trajectory_npy(os.path.join(data_path, 'end_link_poses.npy'), np.array([0.11320721, 0.06791742, 0.04500, 0.70065814, -0.00773338, -0.01030925, 0.71338074]))
    # poses = load_trajectory_npy(os.path.join(data_path, 'end_link_poses.npy'), np.array([0.0758439, -0.00008, 0.0543, 0.703925, -0.00545436867, -0.00678619761, 0.71022]))

    poses = load_trajectory_npy2(pose_npy_file)[::20]
    pcds = [
        rgbd_to_pointcloud(rgb, depth, width, height, camera_matrix)
        for rgb, depth in zip(rgb_full_names[::20], depth_full_names[::20])
    ]
    print(pcds)
    pcds = downsample_pointclouds(pcds, DOWNSAMPLE_VOXEL_SIZE_M)
    pcds = filter_pointclouds(pcds, FILTER_NUM_NEIGHBOR, FILTER_STD_RATIO,
                              FILTER_RADIUS_M)

    estimate_normals(pcds, NORMAL_RADIUS, NORMAL_NUM_NEIGHBOR)

    pcds = np.array(pcds)
    pcds = list(pcds)
    depth_full_names = np.array(depth_full_names)
    depth_full_names = list(depth_full_names)
    rgb_full_names = np.array(rgb_full_names)
    rgb_full_names = list(rgb_full_names)
    print(len(rgb_full_names), len(depth_full_names))
    # assert poses.shape[0] == len(pcds) and len(pcds) == len(
        # rgb_full_names) and len(rgb_full_names) == len(depth_full_names)
    print(poses[len(pcds)-1])
    for i in range(len(pcds)):
        pcds[i].transform(poses[i])
    # transform
    full_pcd = pcds[0]
    for i in range(1, len(pcds)):
        full_pcd = merge_pointclouds(pcds[i], full_pcd)
    ###################################
    # model_pose = np.load('poses.npy')
    # print('model_pose')
    # print(model_pose)
    # model_pcd= o3d.io.read_point_cloud("models/box.ply")
    # # model_pcd.transform(np.linalg.inv(model_pose))
    # model_pcd.transform(model_pose)
    # full_pcd = model_pcd
    # full_pcd = merge_pointclouds(model_pcd,full_pcd)
    # print(len(pcds))
    # merge 
    ###################################
    if visualize:
        o3d.visualization.draw_geometries([full_pcd])

    o3d.io.write_point_cloud('./full_point_cloud.ply',
                          full_pcd,
                          write_ascii=False)
    return full_pcd

def lego_icp_marker(data_path, marker_trans, visualize=False):
    camera_matrix = np.load(os.path.join(data_path, 'camK.npy'))
    markers_transformation = np.load(marker_trans)

    rgb_names = os.listdir(os.path.join(data_path, 'rgb'))
    rgb_names.sort()
    rgb_full_names = [
        os.path.join(data_path, 'rgb/' + name) for name in rgb_names
    ]

    depth_names = os.listdir(os.path.join(data_path, 'depth'))
    depth_names.sort()
    depth_full_names = [
        os.path.join(data_path, 'depth/' + name) for name in depth_names
    ]
    assert len(depth_full_names) == len(rgb_full_names)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    arucoParams = aruco.DetectorParameters_create()
    arucoParams.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    arucoParams.cornerRefinementWinSize = 5

    if visualize:
        print("Estimating poses according to markers...")
        for rgb_name in rgb_full_names:
            rgb = cv2.imread(rgb_name)
            corners, ids, rejected_points = aruco.detectMarkers(
                rgb,
                aruco_dict,
                cameraMatrix=camera_matrix,
                distCoeff=dist_coeffs,
                parameters=arucoParams)
            rvec, tvec, _objPoints = aruco.estimatePoseSingleMarkers(
                corners, MARKER_LENGTH, camera_matrix, dist_coeffs)
            rgb2show = aruco.drawDetectedMarkers(rgb, corners, ids)
            cv2.imshow('Marker', rgb2show)
            cv2.waitKey()
        cv2.destroyAllWindows()
    if INFO:
        print("Processing point clouds...")
    pcds = [
        rgbd_to_pointcloud(rgb, depth, width, height, camera_matrix)
        for rgb, depth in zip(rgb_full_names, depth_full_names)
    ]

    pcds = downsample_pointclouds(pcds, DOWNSAMPLE_VOXEL_SIZE_M)
    pcds = filter_pointclouds(pcds, FILTER_NUM_NEIGHBOR, FILTER_STD_RATIO,
                              FILTER_RADIUS_M)

    pcds = np.array(pcds)
    pcds = list(pcds)
    depth_full_names = np.array(depth_full_names)
    depth_full_names = list(depth_full_names)
    rgb_full_names = np.array(rgb_full_names)
    rgb_full_names = list(rgb_full_names)

    assert len(pcds) == len(rgb_full_names) and len(rgb_full_names) == len(
        depth_full_names)

    poses = get_poses([rgb_full_names[0]], camera_matrix,
                      markers_transformation)
    full_pcd = pcds[len(pcds) - 1]
    for i in range(len(depth_full_names) - 1, 0, -1):

        rgb_1 = cv2.imread(rgb_full_names[i])
        rgb_2 = cv2.imread(rgb_full_names[i - 1])

        corners_1, ids_1, rejected_points_1 = aruco.detectMarkers(
            rgb_1,
            aruco_dict,
            cameraMatrix=camera_matrix,
            distCoeff=dist_coeffs,
            parameters=arucoParams)
        corners_2, ids_2, rejected_points_2 = aruco.detectMarkers(
            rgb_2,
            aruco_dict,
            cameraMatrix=camera_matrix,
            distCoeff=dist_coeffs,
            parameters=arucoParams)

        rvec_1, tvec_1, _objPoints_1 = aruco.estimatePoseSingleMarkers(
            corners_1, MARKER_LENGTH, camera_matrix, dist_coeffs)
        rvec_2, tvec_2, _objPoints_2 = aruco.estimatePoseSingleMarkers(
            corners_2, MARKER_LENGTH, camera_matrix, dist_coeffs)
        tvec_1 = tvec_1 / 1000
        tvec_2 = tvec_2 / 1000
        ids_1 = ids_1.reshape(ids_1.shape[0], ).tolist()
        ids_2 = ids_2.reshape(ids_2.shape[0], ).tolist()
        trans = []
        common_marker = [x for x in ids_1 if x in ids_2]
        if INFO:
            print(common_marker)
        for j in common_marker:
            trans_1 = R.from_rotvec(rvec_1[ids_1.index(j)].reshape(
                3, )).as_dcm()
            trans_2 = R.from_rotvec(rvec_2[ids_2.index(j)].reshape(
                3, )).as_dcm()

            trans_1 = np.vstack((np.hstack(
                (trans_1, tvec_1[ids_1.index(j)].reshape(3, 1))),
                                 np.array([0, 0, 0, 1])))
            trans_2 = np.vstack((np.hstack(
                (trans_2, tvec_2[ids_2.index(j)].reshape(3, 1))),
                                 np.array([0, 0, 0, 1])))
            if INFO:
                print(trans_2)
                print(trans_1)
            trans.append(np.dot(trans_2, np.linalg.inv(trans_1)))
        trans_average = average_transformation(trans)

        full_pcd.transform(trans_average)
        full_pcd = merge_pointclouds(pcds[i - 1], full_pcd)

    full_pcd.transform(poses[0])
    remove_ground_plane(full_pcd)
    if visualize:
        o3d.visualization.draw_geometries([full_pcd])

    o3d.io.write_point_cloud('./full_point_cloud.ply',
                          full_pcd,
                          write_ascii=False)

    return full_pcd


def get_poses(rgb_full_names, camera_matrix, markers_transformation):
    # Get the transformation from camera_frame to 0_frame
    poses = []
    for rgb_name in rgb_full_names:
        rgb = cv2.imread(rgb_name)
        arucoParams = aruco.DetectorParameters_create()
        arucoParams.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        arucoParams.cornerRefinementWinSize = 5
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        corners, ids, rejected_points = aruco.detectMarkers(
            rgb,
            aruco_dict,
            cameraMatrix=camera_matrix,
            distCoeff=dist_coeffs,
            parameters=arucoParams)
        rvec, tvec, _objPoints = aruco.estimatePoseSingleMarkers(
            corners, MARKER_LENGTH, camera_matrix, dist_coeffs)
        rgb2show = aruco.drawDetectedMarkers(rgb, corners, ids)
        pose_before_average = []

        for r, t, idx in zip(rvec, tvec, ids):
            rgb2show = aruco.drawAxis(rgb2show, camera_matrix, dist_coeffs, r,
                                      t, 100)
            trans_mtx = R.from_rotvec(r.reshape(3, )).as_dcm()
            t = t / 1000
            trans_mtx = np.hstack((trans_mtx, t.reshape(3, 1)))
            trans_mtx = np.vstack((trans_mtx, np.array([0, 0, 0, 1])))
            if idx[0] > len(markers_transformation) - 1:
                continue
            camera_pose = np.dot(markers_transformation[idx[0]],
                                 np.linalg.inv(trans_mtx))
            pose_before_average.append(camera_pose)

        poses.append(average_transformation(pose_before_average))
    return np.array(poses)


def average_transformation(pose_before_average):
    quaternion = []
    transl = []
    for mtx in pose_before_average:
        q = Quaternion(matrix=mtx[0:3, 0:3])
        quaternion.append([q[0], q[1], q[2], q[3]])
        transl.append(mtx[0:3, 3].tolist())
    quaternion = np.array(quaternion)
    transl = np.array(transl)
    average_q = averageQuaternions(quaternion)
    average_q = Quaternion(average_q[0], average_q[1], average_q[2],
                           average_q[3])
    average_transl = np.sum(transl, axis=0) / (transl.shape[0])
    average_transformation = average_q.transformation_matrix
    average_transformation[0:3, 3] = average_transl
    return average_transformation


def integrate_pointclouds(full_pcd, new_pcd):
    transform, information = pairwise_registration(full_pcd, new_pcd,
                                                   np.eye(4))
    new_pcd.transform(np.linalg.inv(transform))
    full_new_pcd = merge_pointclouds(full_pcd, new_pcd)
    return full_new_pcd


def merge_pointclouds(pcd1, pcd2):
    merged_points = np.vstack(
        (np.asarray(pcd1.points), np.asarray(pcd2.points)))
    merged_colors = np.vstack(
        (np.asarray(pcd1.colors), np.asarray(pcd2.colors)))

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
    merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

    merged_pcd = merged_pcd.voxel_down_sample(DOWNSAMPLE_VOXEL_SIZE_M)

    return merged_pcd


def pairwise_registration(source, target, init_transform):
    if INFO:
        print("Apply point-to-plane ICP")
    estimate_normals([source, target], NORMAL_RADIUS, NORMAL_NUM_NEIGHBOR)
    icp_coarse = o3d.registration.registration_icp(
        source, target, ICP_MAX_DISTANCE_COARSE, init_transform,
        o3d.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.registration.registration_icp(
        source, target, ICP_MAX_DISTANCE_FINE, icp_coarse.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    # iter = 5
    # icp_fine = o3d.registration.registration_colored_icp(
    #     source, target, 0.001, init_transform,
    #     o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
    #                                             relative_rmse=1e-6,
    #                                             max_iteration=iter))
    transformation_icp = icp_fine.transformation
    information_icp = o3d.registration.get_information_matrix_from_point_clouds(
        source, target, ICP_MAX_DISTANCE_FINE, icp_fine.transformation)
    return transformation_icp, information_icp


def bruteforce_registration(pcds, init_poses):
    if INFO:
        print('Apply brute-force registration')
    pose_graph = o3d.registration.PoseGraph()
    n_pcds = len(pcds)
    for i in range(n_pcds):
        pose_graph.nodes.append(o3d.registration.PoseGraphNode(init_poses[i]))

    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id],
                init_poses[target_id] @ np.linalg.inv(init_poses[source_id]))
            pose_graph.edges.append(
                o3d.registration.PoseGraphEdge(source_id,
                                               target_id,
                                               transformation_icp,
                                               information_icp,
                                               uncertain=True))
    return pose_graph


def odometry_registration(pcds, init_poses, loop_closure=True):
    pose_graph = o3d.registration.PoseGraph()
    odometry = init_poses[0]
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds - 1):
        target_id = source_id + 1
        transformation_icp, information_icp = pairwise_registration(
            pcds[source_id], pcds[target_id],
            init_poses[target_id] @ np.linalg.inv(init_poses[source_id]))

        odometry = np.dot(transformation_icp, odometry)
        pose_graph.nodes.append(
            o3d.registration.PoseGraphNode(np.linalg.inv(odometry)))
        pose_graph.edges.append(
            o3d.registration.PoseGraphEdge(source_id,
                                           target_id,
                                           transformation_icp,
                                           information_icp,
                                           uncertain=False))

    if loop_closure:
        source_id = n_pcds - 1
        target_id = 0
        transformation_icp, information_icp = pairwise_registration(
            pcds[source_id], pcds[target_id],
            init_poses[target_id] @ np.linalg.inv(init_poses[source_id]))
        pose_graph.edges.append(
            o3d.registration.PoseGraphEdge(source_id,
                                           target_id,
                                           transformation_icp,
                                           information_icp,
                                           uncertain=True))
    return pose_graph


def get_plane(pcds, poses, camera_matrix):
    all_masks = []
    all_points = []

    for pcd, pose in zip(pcds, poses):
        mask_array = np.zeros((1280, 720))
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        colors = np.asarray(pcd.colors)

        mask1 = points[:, 2] > PLANE_Z_MIN
        mask2 = points[:, 2] < PLANE_Z_MAX
        mask3 = points[:, 1] > PLANE_Y_MIN
        mask4 = points[:, 1] < PLANE_Y_MAX
        mask5 = points[:, 0] > PLANE_X_MIN
        mask6 = points[:, 0] < PLANE_X_MAX
        mask = []
        for i, j, k, l, m, n in zip(mask1.tolist(), mask2.tolist(),
                                    mask3.tolist(), mask4.tolist(),
                                    mask5.tolist(), mask6.tolist()):
            if i and j and k and l and m and n:
                mask.append(True)
            else:
                mask.append(False)

        mask = np.array(mask)
        new_points = points[mask, ...]
        new_normals = normals[mask, ...]
        new_colors = colors[mask, ...]

        for i in range(new_points.shape[0]):
            a = np.dot(
                np.linalg.inv(pose),
                np.array(
                    [new_points[i, 0], new_points[i, 1], new_points[i, 2], 1]))
            x = a[0]
            y = a[1]
            z = a[2]
            px = int(x * camera_matrix[0, 0] / z + camera_matrix[0, 2])
            py = int(y * camera_matrix[1, 1] / z + camera_matrix[1, 2])
            mask_array[px, py] = 1

        mask_array = mask_array.T

        all_masks.append(mask_array)
        all_points.append(new_points)
    return np.array(all_masks), np.array(all_points)


def remove_ground_plane(pcd):
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    colors = np.asarray(pcd.colors)

    pcd.points = o3d.Vector3dVector(points)
    pcd.colors = o3d.Vector3dVector(colors)


def filter_pointclouds(pcds, num_neighbors, std_ratio, radius):
    for i in range(len(pcds)):
        cl, ind = pcds[i].remove_statistical_outlier(
                                                  nb_neighbors=num_neighbors,
                                                  std_ratio=radius)
        pcds[i] = o3d.geometry.PointCloud.select_down_sample(pcds[i], ind)
        if radius > 0:
            cl, ind = pcds[i].remove_statistical_outlier(
                                                 nb_neighbors=num_neighbors,
                                                 std_ratio=radius)
            pcds[i] = o3d.geometry.PointCloud.select_down_sample(pcds[i], ind)

    return pcds


def downsample_pointclouds(pcds, voxel_size):
    for pcd in pcds:
        pcd.voxel_down_sample(voxel_size=voxel_size)
    # down_pcds = [
    #     pcd.voxel_down_sample(pcd, voxel_size=voxel_size)
    #     for pcd in pcds
    # ]
    return pcds


def estimate_normals(pcds, radius, max_nn):
    for pcd in pcds:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                 radius=radius, max_nn=max_nn))


def rgbd_to_pointcloud(color_image_name, depth_image_name, width, height,
                       camera_matrix):
    color = o3d.io.read_image(color_image_name)

    depth = cv2.imread(depth_image_name, cv2.IMREAD_UNCHANGED).astype(np.float32)
    # depth += 15.0 # add 15 as a hot fix since current reconstructed lego is a bit wider.
    depth /= 1000.0
    depth[depth < MIN_DEPTH_M] = 0
    depth[depth > MAX_DEPTH_M] = 0

    rgbd_image = o3d.geometry.RGBDImage()
    rgbd_image.color = color
    rgbd_image.depth = o3d.geometry.Image(depth)

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, camera_matrix[0, 0],
                             camera_matrix[1, 1], camera_matrix[0, 2],
                             camera_matrix[1, 2])
    if INFO:
        print(o3d.geometry.create_point_cloud_from_rgbd_image)
    # pcd = o3d.geometry.create_point_cloud_from_rgbd_image(
    #     rgbd_image, intrinsic, np.eye(4))
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, np.eye(4))

    return pcd


def load_trajectory_npy2(trajectory_npy):
    traj = np.load(trajectory_npy)
    Tbe = np.zeros((traj.shape[0], 4, 4))
    # for i in range(traj.shape[0]):
        # t = traj[i, :3]
        # q = traj[i, 3:]
        # Tbe[i, :3, :3] = quat2mat(q)
        # Tbe[i, :3, 3] = t
        # Tbe[i, 3, 3] = 1.0
    return traj

def load_trajectory_npy(trajectory_npy, cam_rel):
    camP = np.zeros((4, 4))
    camP[:3, :3] = quat2mat(cam_rel[3:])
    camP[:3, 3] = cam_rel[:3]
    camP[3, 3] = 1.0
    traj = np.load(trajectory_npy)
    Tbe = np.zeros((traj.shape[0], 4, 4))
    for i in range(traj.shape[0]):
        t = traj[i, :3]
        q = traj[i, 3:]
        Tbe[i, :3, :3] = quat2mat(q)
        Tbe[i, :3, 3] = t
        Tbe[i, 3, 3] = 1.0
        Tbe[i] = np.dot(Tbe[i],camP)
    return Tbe

if __name__ == '__main__':
    lego_icp('./scene_31/data/')
    # vis('./scene_01/data/')

    # camera_matrix = np.load(os.path.join('./scene_00/data_kinect/', 'camK.npy'))
    # print(camera_matrix)
