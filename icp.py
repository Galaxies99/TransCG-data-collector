from trans3d import get_mat, get_pose
import open3d as o3d
import numpy as np
import cv2
import os
import copy
# from legoICP_marker import legoICP_marker

MAX_DEPTH_M = 0.5
MIN_DEPTH_M = 0.2

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

DOWNSAMPLE_VOXEL_SIZE_M = 0.005

def read_cloud(rgb_image_path,depth_image_path,cam):
    down_pcd = rgbd_to_pointcloud(rgb_image_path, depth_image_path, IMAGE_WIDTH, IMAGE_HEIGHT, cam)
    o3d.io.write_point_cloud('./full_point_cloud.ply', down_pcd, write_ascii=True)
    return down_pcd

def rgbd_to_pointcloud(color_image_name, depth_image_name, width, height, camera_matrix):
    color = o3d.io.read_image(color_image_name)
    depth = cv2.imread(depth_image_name, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth /= 1000.0
    depth[depth < MIN_DEPTH_M] = 0
    depth[depth > MAX_DEPTH_M] = 0

    rgbd_image = o3d.geometry.RGBDImage()
    rgbd_image.color = color
    rgbd_image.depth = o3d.geometry.Image(depth)

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width, height, camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2])
    pcd = o3d.geometry.PointCloud()
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, np.eye(4))
    # pcd = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd_image, intrinsic, np.eye(4))
    return pcd

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def refine_6d_pose(rgb_image_path,depth_image_path, scene, model, cam, x, y ,z, alpha, beta, gamma):
    # point cloud build
    # scene = read_cloud(rgb_image_path = rgb_image_path,depth_image_path = depth_image_path,cam=cam)
    # 6d -> matrix
    current_transformation = get_mat(x, y ,z, alpha, beta, gamma)
    # icp
    # print("log:downsample")
    # model_down = o3d.geometry.voxel_down_sample(model, DOWNSAMPLE_VOXEL_SIZE_M)
    # scene_down = o3d.geometry.voxel_down_sample(scene, DOWNSAMPLE_VOXEL_SIZE_M)
    model_down=model
    # model_down.voxel_down_sample(DOWNSAMPLE_VOXEL_SIZE_M)
    scene_down = scene
    # scene_down.voxel_down_sample(DOWNSAMPLE_VOXEL_SIZE_M)
    radius_normal = DOWNSAMPLE_VOXEL_SIZE_M * 2
    # o3d.geometry.estimate_normals(model, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))
    # o3d.geometry.estimate_normals(scene, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))
    model_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))
    scene_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))

    radius_feature = DOWNSAMPLE_VOXEL_SIZE_M * 5
    model_fpfh = o3d.registration.compute_fpfh_feature(model_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    scene_fpfh = o3d.registration.compute_fpfh_feature(scene_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    print("log:icp")
    # draw_registration_result(model, scene_down, current_transformation)
    result_icp = o3d.registration.registration_icp(
            model_down, scene_down, DOWNSAMPLE_VOXEL_SIZE_M, current_transformation,
            o3d.registration.TransformationEstimationPointToPoint())
    o3d.io.write_point_cloud('./scene_trans.ply', scene_down, write_ascii=True)
    o3d.io.write_point_cloud('./model.ply', copy.deepcopy(model).transform(current_transformation), write_ascii=True)
    # result_icp = o3d.registration.registration_colored_icp(
    #     model, scene, 0.02, current_transformation,
    #     o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
    #                                             relative_rmse=1e-6,
    #                                             max_iteration=30))
    # draw_registration_result(model, scene_down, result_icp.transformation)
    print("log:icp done")
    # matrix -> 6d
    x,y,z, alpha, beta, gamma = get_pose(result_icp.transformation)
    return x,y,z, alpha, beta, gamma

