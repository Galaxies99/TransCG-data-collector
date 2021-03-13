import open3d as o3d
import numpy as np
from transforms3d.quaternions import quat2mat, mat2quat
import yaml
import cv2 as cv
import os
from cv2 import aruco
from scipy.spatial.transform import Rotation as R
from averageQuaternions import averageQuaternions
from pyquaternion import Quaternion
from umeyama import umeyama_alignment

MAX_DEPTH_M = 0.9
MIN_DEPTH_M = 0.2
DOWNSAMPLE_VOXEL_SIZE_M = 0.003

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
dist_coeffs = np.array([0., 0., 0., 0.]).reshape(4,1)
# dist_coeffs = np.array([0.30, -0.33, 0.005, -0.001]).reshape(4, 1)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)


def legoICP(DATA_PATH, markerTrans):
    camera_matrix = np.load(os.path.join(DATA_PATH, 'camK.npy'))
    markers_transformation = np.load(markerTrans)

    rgb_names = os.listdir(os.path.join(DATA_PATH, 'rgb'))
    rgb_names.sort()
    rgb_full_names = [os.path.join(DATA_PATH, 'rgb/' + name) for name in rgb_names]

    depth_names = os.listdir(os.path.join(DATA_PATH, 'depth'))
    depth_names.sort()
    depth_full_names = [os.path.join(DATA_PATH, 'depth/' + name) for name in depth_names]
    # assert len(depth_full_names) == len(rgb_full_names)
    
    poses = get_poses(rgb_full_names, camera_matrix, markers_transformation)
    pcds = [rgbd_to_pointcloud(rgb, depth, width, height, camera_matrix)
            for rgb, depth in zip(rgb_full_names, depth_full_names)]
    
    pcds = downsample_pointclouds(pcds, DOWNSAMPLE_VOXEL_SIZE_M)
    pcds = filter_pointclouds(pcds, FILTER_NUM_NEIGHBOR, FILTER_STD_RATIO, FILTER_RADIUS_M)

    estimate_normals(pcds, NORMAL_RADIUS, NORMAL_NUM_NEIGHBOR)

    pcds = np.array(pcds)
    pcds = list(pcds)
    depth_full_names = np.array(depth_full_names)
    depth_full_names = list(depth_full_names)
    rgb_full_names = np.array(rgb_full_names)
    rgb_full_names = list(rgb_full_names)

    assert poses.shape[0] == len(pcds) and len(pcds) == len(rgb_full_names) and len(rgb_full_names) == len(depth_full_names)

    trans_all = []
    for i in range(len(pcds)):
        pcds[i].transform(poses[i])
        trans_all.append(np.dot(poses[0], np.linalg.inv(poses[i])))
        # points = np.asarray(pcds[i].points)
        # colors = np.asarray(pcds[i].colors)
        # new_pcd = o3d.geometry.PointCloud()
        # index = np.bitwise_and(np.bitwise_and(0.02 < points[:, 0], points[:, 0] < 0.30),
        #                        np.bitwise_and(-0.30 < points[:, 1], points[:, 1] < -0.02))
        # index = np.bitwise_and(index, points[:, 2] > 0.015 - 0.03)
        # new_pcd.points = o3d.utility.Vector3dVector(points[index])
        # new_pcd.colors = o3d.utility.Vector3dVector(colors[index])
        # pcds[i] = new_pcd
    np.save("cam_poses_new.npy", np.array(trans_all))
        
    #remove_ground_plane(pcds)
    #full_pcd = pointclouds_adjustment(rgb_full_names,depth_full_names,poses,pcds,camera_matrix)
    full_pcd = pcds[0]
    for i in range(1, len(pcds)):
        full_pcd = merge_pointclouds(pcds[i],full_pcd)

    
    o3d.write_point_cloud('./full_point_cloud.ply', full_pcd, write_ascii=False)
    o3d.draw_geometries([full_pcd])
    return full_pcd

    # pose_graph = bruteforce_registration(pcds, poses)
    # pose_graph = odometry_registration(pcds, poses, loop_closure=False)
    #
    # print("Optimizing PoseGraph ...")
    # option = o3d.registration.GlobalOptimizationOption(
    #     max_correspondence_distance=ICP_MAX_DISTANCE_FINE,
    #     edge_prune_threshold=0.25,
    #     reference_node=0)
    # o3d.registration.global_optimization(
    #     pose_graph, o3d.registration.GlobalOptimizationLevenbergMarquardt(),
    #     o3d.registration.GlobalOptimizationConvergenceCriteria(), option)
    #
    # for i in range(len(pcds)):
    #     pcds[i].transform(pose_graph.nodes[i].pose)
    #
    # o3d.draw_geometries(pcds)

def get_poses(rgb_full_names, camera_matrix, markers_transformation):
    # Get the transformation from camera_frame to 0_frame
    poses = []
    for rgb_name in rgb_full_names:
        rgb = cv.imread(rgb_name)
        arucoParams = aruco.DetectorParameters_create()
        arucoParams.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
        arucoParams.cornerRefinementWinSize = 5
        corners, ids, rejected_points = aruco.detectMarkers(
            rgb, aruco_dict, cameraMatrix=camera_matrix, distCoeff=dist_coeffs, parameters = arucoParams)
        rvec, tvec, _objPoints = aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)
        rgb2show = aruco.drawDetectedMarkers(rgb, corners, ids)
        pose_before_average = []
        
        for r,t,idx in zip(rvec,tvec, ids):
            if idx[0] > 7:
                continue
            rgb2show = aruco.drawAxis(rgb2show, camera_matrix, dist_coeffs, r, t, 100)
            trans_mtx = R.from_rotvec(r.reshape(3,)).as_dcm()
            t = t/1000
            trans_mtx = np.hstack((trans_mtx, t.reshape(3,1)))
            trans_mtx = np.vstack((trans_mtx, np.array([0,0,0,1])))
            camera_pose = np.dot(markers_transformation[idx[0]], np.linalg.inv(trans_mtx))
            pose_before_average.append(camera_pose)
        
        poses.append(average_transformation(pose_before_average))
        # cv.imshow('Marker', rgb2show)
        # cv.waitKey()
    # cv.destroyAllWindows()
    return np.array(poses)

def pointclouds_adjustment(rgb_full_names, depth_full_names,poses, pcds,camera_matrix):
    # Make adjustment on the point clouds
    camera_mtx = np.hstack((camera_matrix,np.array([0,0,0]).reshape(3,1)))
    camera_mtx = np.vstack((camera_mtx, np.array([0,0,0,1])))

    first = True
    for i,j,p,pcd in zip(rgb_full_names,depth_full_names,poses,pcds):
        source = []
        target = []
        rgb = cv.imread(i)
        corners, ids, rejected_points = aruco.detectMarkers(
            rgb, aruco_dict, cameraMatrix=camera_matrix, distCoeff=dist_coeffs)
        depth = cv.imread(j, cv.IMREAD_UNCHANGED).astype(np.float32)
        ids = ids.reshape(ids.shape[0],).tolist()

        index = 0
        for pixel in corners:
            pixel = np.sum(pixel.reshape(4,2),axis=0)
            u = pixel[0]/4
            v = pixel[1]/4
            d = 0
            for aa in range(-1,2,1):
                for bb in range(-1,2,1):
                    d = d + depth[int(v+aa),int(u+bb)]/1000
            d = d/9
            if d == 0:
                del ids[index]
                index += 1
                continue
            index += 1
            source.append([u*d,v*d,d,1])
        source = np.array(source).T
        source = np.dot(np.linalg.inv(camera_mtx),source)
        source = np.dot(p,source)    
        source = source[0:3,:]

        for index in ids:
            target.append(markers_transformation[index,0:3,3].tolist())     
        target = np.array(target).T

        delete = []
        for i in range(source.shape[1]):
            if max(abs(source[:,i]-target[:,i]))>0.08 or abs(source[2,i]-target[2,i])>0.05:
                delete.append(i)
        # # for i in delete:
        # source = np.delete(source,np.array(delete),1)
        # target = np.delete(target,np.array(delete),1)
    
        assert source.shape[0] == target.shape[0] and source.shape[1]==target.shape[1]
        if source.shape[1] > 2:
            r,t,c = umeyama_alignment(source, target)
            transform = np.hstack((r,t.reshape(3,1)))
            transform = np.vstack((transform, np.array([0,0,0,1])))
        else:
            tmp = target - source
            transl = np.sum(tmp, axis=1)/tmp.shape[1]
            transform = np.eye(4,4)
            transform[0:3,3] = transl
            
        pcd.transform(transform)
        if first == True:
            full_pcd = pcd
            first = False
        else:
            full_pcd = merge_pointclouds(pcd,full_pcd)
        o3d.visualization.draw_geometries([full_pcd])

    return full_pcd


def average_transformation(pose_before_average):
    quaternion = []
    transl = []
    for mtx in pose_before_average: 
        q = Quaternion(matrix=mtx[0:3, 0:3])
        quaternion.append([q[0],q[1],q[2],q[3]])
        transl.append(mtx[0:3,3].tolist())
    quaternion = np.array(quaternion)
    transl = np.array(transl)
    average_q = averageQuaternions(quaternion)
    average_q = Quaternion(average_q[0],average_q[1],average_q[2],average_q[3])
    average_transl = np.sum(transl,axis=0)/(transl.shape[0])
    average_transformation = average_q.transformation_matrix
    average_transformation[0:3,3] = average_transl
    return average_transformation

    
def integrate_pointclouds(full_pcd, new_pcd):
    transform, information = pairwise_registration(full_pcd, new_pcd, np.eye(4))
    new_pcd.transform(np.linalg.inv(transform))
    full_new_pcd = merge_pointclouds(full_pcd, new_pcd)
    return full_new_pcd


def merge_pointclouds(pcd1, pcd2):
    merged_points = np.vstack((np.asarray(pcd1.points), np.asarray(pcd2.points)))
    merged_colors = np.vstack((np.asarray(pcd1.colors), np.asarray(pcd2.colors)))
    # merged_normals = np.vstack((np.asarray(pcd1.normals), np.asarray(pcd2.normals)))

    merged_pcd = o3d.PointCloud()
    merged_pcd.points = o3d.Vector3dVector(merged_points)
    merged_pcd.colors = o3d.Vector3dVector(merged_colors)
    # merged_pcd.normals = o3d.Vector3dVector(merged_normals)

    merged_pcd = o3d.voxel_down_sample(merged_pcd, DOWNSAMPLE_VOXEL_SIZE_M)

    cl, ind = o3d.statistical_outlier_removal(merged_pcd, nb_neighbors=FILTER_NUM_NEIGHBOR, std_ratio=FILTER_STD_RATIO)
    merged_pcd = o3d.select_down_sample(merged_pcd, ind)

    o3d.estimate_normals(merged_pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS, max_nn=NORMAL_NUM_NEIGHBOR))

    return merged_pcd


def pairwise_registration(source, target, init_transform):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.registration.registration_icp(
        source, target, ICP_MAX_DISTANCE_COARSE, init_transform,
        o3d.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.registration.registration_icp(
        source, target, ICP_MAX_DISTANCE_FINE,
        icp_coarse.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.registration.get_information_matrix_from_point_clouds(
        source, target, ICP_MAX_DISTANCE_FINE,
        icp_fine.transformation)
    return transformation_icp, information_icp


def bruteforce_registration(pcds, init_poses):
    print('Apply brute-force registration')
    pose_graph = o3d.registration.PoseGraph()
    n_pcds = len(pcds)
    for i in range(n_pcds):
        pose_graph.nodes.append(o3d.registration.PoseGraphNode(init_poses[i]))

    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], init_poses[target_id] @ np.linalg.inv(init_poses[source_id]))
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
            pcds[source_id], pcds[target_id], init_poses[target_id] @ np.linalg.inv(init_poses[source_id]))

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
            pcds[source_id], pcds[target_id], init_poses[target_id] @ np.linalg.inv(init_poses[source_id]))
        pose_graph.edges.append(
            o3d.registration.PoseGraphEdge(source_id,
                                           target_id,
                                           transformation_icp,
                                           information_icp,
                                           uncertain=True))
    return pose_graph

def get_plane(pcds,poses,camera_matrix):
    all_masks = []
    all_points = []

    for pcd, pose in zip(pcds, poses):
        mask_array = np.zeros((1280,720)) 
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        colors = np.asarray(pcd.colors)

        mask1 = points[:,2] > PLANE_Z_MIN
        mask2 = points[:,2] < PLANE_Z_MAX
        mask3 = points[:,1] > PLANE_Y_MIN
        mask4 = points[:,1] < PLANE_Y_MAX
        mask5 = points[:,0] > PLANE_X_MIN
        mask6 = points[:,0] < PLANE_X_MAX
        mask = []
        for i,j,k,l,m,n in zip(mask1.tolist(),mask2.tolist(),mask3.tolist(),mask4.tolist(),mask5.tolist(),mask6.tolist()):
            if i and j and k and l and m and n:
                mask.append(True)
            else:
                mask.append(False)

        mask = np.array(mask)
        new_points = points[mask, ...]  
        new_normals = normals[mask, ...] 
        new_colors = colors[mask, ...]
                     
        for i in range(new_points.shape[0]):
            a = np.dot(np.linalg.inv(pose),np.array([new_points[i,0],new_points[i,1],new_points[i,2],1]))
            x = a[0]
            y = a[1]
            z = a[2]
            px = int(x*camera_matrix[0,0]/z + camera_matrix[0,2])
            py = int(y*camera_matrix[1,1]/z + camera_matrix[1,2])
            mask_array[px,py] = 1
        
        mask_array = mask_array.T
        #cv.imshow("gray", mask_array)
    
        all_masks.append(mask_array)
        all_points.append(new_points)
    return np.array(all_masks), np.array(all_points)
    


def remove_ground_plane(pcds):
    for pcd in pcds:
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        colors = np.asarray(pcd.colors)

        mask = points[:, 2] > 0.046
        
        new_points = points[mask, ...] 
        new_normals = normals[mask, ...] 
        new_colors = colors[mask, ...]

        pcd.points = o3d.Vector3dVector(new_points)
        pcd.normals = o3d.Vector3dVector(new_normals)
        pcd.colors = o3d.Vector3dVector(new_colors)


def filter_pointclouds(pcds, num_neighbors, std_ratio, radius):
    for i in range(len(pcds)):
        cl, ind = o3d.statistical_outlier_removal(pcds[i], nb_neighbors=num_neighbors, std_ratio=std_ratio)
        pcds[i] = o3d.select_down_sample(pcds[i], ind)
        if radius > 0:
            cl, ind = o3d.radius_outlier_removal(pcds[i], nb_points=num_neighbors, radius=radius)
            pcds[i] = o3d.select_down_sample(pcds[i], ind)

    return pcds


def downsample_pointclouds(pcds, voxel_size):
    down_pcds = [o3d.geometry.voxel_down_sample(pcd, voxel_size=voxel_size) for pcd in pcds]
    return down_pcds


def estimate_normals(pcds, radius, max_nn):
    for pcd in pcds:
        o3d.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=max_nn))


def rgbd_to_pointcloud(color_image_name, depth_image_name, width, height, camera_matrix):
    color = o3d.io.read_image(color_image_name)

    depth = cv.imread(depth_image_name, cv.IMREAD_UNCHANGED).astype(np.float32)
    depth += 15.0
    depth /= 1000.0  # from millimeters to meters
    depth[depth < MIN_DEPTH_M] = 0
    depth[depth > MAX_DEPTH_M] = 0

    rgbd_image = o3d.RGBDImage()
    rgbd_image.color = color
    rgbd_image.depth = o3d.Image(depth)

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width, height, camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2])
    pcd = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd_image, intrinsic, np.eye(4))
    # o3d.visualization.draw_geometries([pcd])
    return pcd


def load_trajectory_npy(trajectory_npy):
    traj = np.load(trajectory_npy)
    Tbe = np.zeros((traj.shape[0], 4, 4))
    for i in range(traj.shape[0]):
        t = traj[i, :3]
        q = traj[i, 3:]
        Tbe[i, :3, :3] = quat2mat(q)
        Tbe[i, :3, 3] = t
        Tbe[i, 3, 3] = 1.0
    return Tbe


def load_camera_parameters(yaml_name):
    try:
        f = open(yaml_name)
        next(f)  # skip the first line
    except UnicodeDecodeError:
        f = open(yaml_name, encoding='UTF-8')
        next(f)

    content = yaml.load(f)
    # fx 0  cx
    # 0  fy cy
    # 0  0  1
    camera_matrix = np.eye(3)
    camera_matrix[0, 0] = content['Camera.fx']
    camera_matrix[0, 2] = content['Camera.cx']
    camera_matrix[1, 1] = content['Camera.fy']
    camera_matrix[1, 2] = content['Camera.cy']

    dist_coeffs = np.zeros(4)
    dist_coeffs[0] = content['Camera.k1']
    dist_coeffs[1] = content['Camera.k2']
    dist_coeffs[2] = content['Camera.p1']
    dist_coeffs[3] = content['Camera.p2']

    width = content['Camera.w']
    height = content['Camera.h']

    f.close()
    return camera_matrix, dist_coeffs, width, height


if __name__ == '__main__':
    legoICP('../test_mega3')
