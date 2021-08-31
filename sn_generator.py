import os
import cv2
import json
import argparse
import open3d as o3d
import numpy as np


class SurfaceNormalGenerator(object):
    """
    Surface normal generator for the dataset: given an depth image, generate its surface normal.
    """
    def __init__(self, data_path, **kwargs):
        super(SurfaceNormalGenerator, self).__init__()
        self.data_path = data_path
        self.fault_depth_limit = kwargs.get('fault_depth_limit', 0.2)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.display_log = kwargs.get('display_log', False)
        self._load_camera_intrinsics()
    
    def _point_cloud(self, color, depth, camera_intrinsics, use_mask = False, use_inpainting = True, scale = 1000.0, inpainting_radius = 5):
        """
        Given the depth image, return the point cloud in open3d format.
        The code is adapted from [graspnet.py] in the [graspnetAPI] repository.
        """
        d = depth.copy()
        c = color.copy() / 255.0
        
        if use_inpainting:
            fault_mask = (d < self.fault_depth_limit * scale)
            d[fault_mask] = 0
            inpainting_mask = (np.abs(d) < self.epsilon * scale).astype(np.uint8)  
            d = cv2.inpaint(d, inpainting_mask, inpainting_radius, cv2.INPAINT_NS)

        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

        xmap, ymap = np.arange(d.shape[1]), np.arange(d.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)

        points_z = d / scale
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z
        points = np.stack([points_x, points_y, points_z], axis = -1)

        if use_mask:
            mask = (points_z > 0)
            points = points[mask]
            c = c[mask]
        else:
            points = points.reshape((-1, 3))
            c = c.reshape((-1, 3))
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(c)
        return cloud

    def gen_normals(self, color, depth, camera_intrinsics, scale = 1000.0):
        """
        Generate surface normal using the given depth image.
        The code is adapted from [gen_normals.py] in the [rgbd_graspnet] repository.
        """
        cur_radius = 5
        while True:
            pcd = self._point_cloud(color, depth, camera_intrinsics, scale = scale, inpainting_radius = cur_radius)
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(150))
            normals = np.asarray(pcd.normals)
            if normals.shape[0] == depth.shape[0] * depth.shape[1]:
                break
            else:
                if self.display_log:
                    print('[Log] Inpainting with radius {} failed, try {} again.'.format(cur_radius, cur_radius * 2))
                cur_radius = cur_radius * 2
            if cur_radius >= 720:
                raise RuntimeError('Unable to generate the surface normal!')
        normal = normals.reshape((depth.shape[0], depth.shape[1], 3))
        normal = ((normal + 1.0) / 2 * 255.0).astype(np.uint8)
        return normal
    
    def _load_camera_intrinsics(self):
        self.camera_intrinsics = {
            'depth1-gt.png': np.load(os.path.join(self.data_path, 'camera_intrinsics', 'camIntrinsics-D435.npy')),
            'depth2-gt.png': np.load(os.path.join(self.data_path, 'camera_intrinsics', 'camIntrinsics-L515.npy'))
        }
    
    def gen_image_normals(self, image_path, color_name, depth_name, normal_name):
        if self.display_log:
            print('[Log] Generating surface normals for {}'.format(os.path.join(image_path, depth_name)))
        depth = cv2.imread(os.path.join(image_path, depth_name), cv2.IMREAD_UNCHANGED)
        color = cv2.imread(os.path.join(image_path, color_name))
        camera_intrinsics = self.camera_intrinsics[depth_name]
        scale = (1000.0 if "depth1" in depth_name else 4000.0)
        normal = self.gen_normals(color, depth, camera_intrinsics, scale)
        cv2.imwrite(os.path.join(image_path, normal_name), normal)

    def gen_scene_normals(self, scene_id):
        scene_path = os.path.join(self.data_path, "scene{}".format(scene_id))
        with open(os.path.join(scene_path, 'metadata.json'), 'r') as fp:
            scene_metadata = json.load(fp)
        for image_id in scene_metadata['D435_valid_perspective_list']:
            self.gen_image_normals(os.path.join(scene_path, str(image_id)), 'rgb1.png', 'depth1-gt.png', 'depth1-gt-sn.png')
        for image_id in scene_metadata['L515_valid_perspective_list']:
            self.gen_image_normals(os.path.join(scene_path, str(image_id)), 'rgb2.png', 'depth2-gt.png', 'depth2-gt-sn.png')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default = 'data', help = 'data for visualization', type = str)
    parser.add_argument('--begin_id', default = 1, help = 'begin scene id', type = int)
    parser.add_argument('--end_id', default = 130, help = 'end scene id', type = int)
    FLAGS = parser.parse_args()
    sn_gen = SurfaceNormalGenerator(FLAGS.data_dir, display_log = True)
    begin_id = int(FLAGS.begin_id)
    end_id = int(FLAGS.end_id)
    for id in range(begin_id, end_id + 1):
        sn_gen.gen_scene_normals(id)
