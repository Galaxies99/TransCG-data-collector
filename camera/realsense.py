import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2

class RealSense():
    def __init__(self, frame_rate = 30, resolution = (1280,720), resolution_depth = (1280,720), use_infrared = False):
        '''
        **Input:**

        - frame_rate: int of how many frames to take in one second. Don't exceed the maximum value for each resolution.

        - resolution: tuple of ints, (width, height).

        - use_infrared: bool, optional, default: False, whether to use the infrared camera.
        '''
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, resolution_depth[0], resolution_depth[1], rs.format.z16, frame_rate)
        self.config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, frame_rate)
        self.use_infrared = use_infrared
        if use_infrared:
            self.config.enable_stream(rs.stream.infrared, 1, resolution[0], resolution[1], rs.format.y8, frame_rate)
            self.config.enable_stream(rs.stream.infrared, 2, resolution[0], resolution[1], rs.format.y8, frame_rate)
        self.pipeline.start(self.config)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.colorizer = rs.colorizer()

    def advanced():
        import json

        DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07", "0B3A", "0B5C"]

        
        ctx = rs.context()
        ds5_dev = rs.device()
        dev = ctx.query_devices()[0]
        try:
            advnc_mode = rs.rs400_advanced_mode(dev)
            print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

            # Loop until we successfully enable advanced mode
            while not advnc_mode.is_enabled():
                print("Trying to enable advanced mode...")
                advnc_mode.toggle_advanced_mode(True)
                # At this point the device will disconnect and re-connect.
                print("Sleeping for 5 seconds...")
                time.sleep(5)
                # The 'dev' object will become invalid and we need to initialize it again
                dev = ctx.query_devices()[0]
                advnc_mode = rs.rs400_advanced_mode(dev)
                print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

            # Get each control's current value
            print("Depth Control: \n", advnc_mode.get_depth_control())
            print("RSM: \n", advnc_mode.get_rsm())
            print("RAU Support Vector Control: \n", advnc_mode.get_rau_support_vector_control())
            print("Color Control: \n", advnc_mode.get_color_control())
            print("RAU Thresholds Control: \n", advnc_mode.get_rau_thresholds_control())
            print("SLO Color Thresholds Control: \n", advnc_mode.get_slo_color_thresholds_control())
            print("SLO Penalty Control: \n", advnc_mode.get_slo_penalty_control())
            print("HDAD: \n", advnc_mode.get_hdad())
            print("Color Correction: \n", advnc_mode.get_color_correction())
            print("Depth Table: \n", advnc_mode.get_depth_table())
            print("Auto Exposure Control: \n", advnc_mode.get_ae_control())
            print("Census: \n", advnc_mode.get_census())

            as_json_object = json.loads(serialized_string)

            # We can also load controls from a json string
            # For Python 2, the values in 'as_json_object' dict need to be converted from unicode object to utf-8
            if type(next(iter(as_json_object))) != str:
                as_json_object = {k.encode('utf-8'): v.encode("utf-8") for k, v in as_json_object.items()}
            # The C++ JSON parser requires double-quotes for the json object so we need
            # to replace the single quote of the pythonic json to double-quotes
            json_string = str(as_json_object).replace("'", '\"')
            advnc_mode.load_json(json_string)

        except Exception as e:
            print(e)
            pass

    def get_rgbd_image(self, return_pcd=False, return_color_depth=False):
        '''
        **Input:**

        - return_pcd: bool of whether to return open3d.Pointcloud.

        - return_color_depth: bool of whether to return color_depth_image.

        **Output:**

        - tuple of color_image, depth_image.

        - color_image: np.array of shape (height, width, 3).

        - depth_image: np.array of shape (height, width), dtype = np.uint16.
        '''
        frames = self.pipeline.wait_for_frames()
        frameset = self.align.process(frames)
        depth_frame = frameset.get_depth_frame()
        color_frame = frameset.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        colorized_depth = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())


        if return_pcd:
            colors = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) / 255.0
            depths = depth_image
            param = o3d.camera.PinholeCameraParameters()
            param.intrinsic.set_intrinsics(1280,720,927.17,927.37,651.32,349.62)
            intrinsics = param.intrinsic.intrinsic_matrix
            fx, fy = intrinsics[0,0], intrinsics[1,1]
            cx, cy = intrinsics[0,2], intrinsics[1,2]
            s = 1000.0
            
            xmap, ymap = np.arange(colors.shape[1]), np.arange(colors.shape[0])
            xmap, ymap = np.meshgrid(xmap, ymap)

            points_z = depths / s
            points_x = (xmap - cx) / fx * points_z
            points_y = (ymap - cy) / fy * points_z

            mask = (points_z > 0)
            points = np.stack([points_x, points_y, points_z], axis=-1)
            points = points[mask]
            colors = colors[mask]
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            cloud.colors = o3d.utility.Vector3dVector(colors)
        if not (return_pcd or return_color_depth):
            return color_image, depth_image
        if (not return_color_depth) and return_pcd:
            return color_image, depth_image, cloud
        if (not return_pcd) and return_color_depth:
            return color_image, depth_image, colorized_depth
        if return_color_depth and return_pcd:
            return color_image, depth_image, cloud, colorized_depth

            

    def get_rgb_image(self):
        '''
        **Output:**

        - color_image: np.array of shape (height, width, 3).
        '''
        self.frames = self.pipeline.wait_for_frames()
        color_frame = self.frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        return color_image

    def get_depth_image(self):
        '''
        **Output:**

        - depth_image: np.array of shape (height, width), dtype = np.uint16.
        '''
        self.frames = self.pipeline.wait_for_frames()
        depth_frame = self.frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        return depth_image

    def get_infrared_image(self):
        '''
        **Output:**

        - infrared_image_left, infrared_image_right: np.array of shape (height, width), dtype=np.uint8.
        '''
        if not self.use_infrared:
            return None
        self.frames = self.pipeline.wait_for_frames()
        infrared_frame_left = self.frames.get_infrared_image(1)
        infrared_frame_right = self.frames.get_infrared_image(2)
        infrared_image_left = np.asanyarray(infrared_frame_left.get_data())
        infrared_image_right = np.asanyarray(infrared_frame_right.get_data())
        return infrared_image_left, infrared_image_right

    def get_full_image(self):
        '''
        **Output:**

        - color_image, depth_image
        
        - (optional) infrared_image_left, infrared_image_right when 'use_infrared' is True.
        '''
        if self.use_infrared:
            frames = self.pipeline.wait_for_frames()
            frameset = self.align.process(frames)
            color_image = np.asanyarray(frameset.get_color_frame().get_data())
            depth_image = np.asanyarray(frameset.get_depth_frame().get_data())
            infrared_image_left = np.asanyarray(frameset.get_infrared_frame(1).get_data())
            infrared_image_right = np.asanyarray(frameset.get_infrared_frame(2).get_data())
            return color_image, depth_image, infrared_image_left, infrared_image_right
        else:
            frames = self.pipeline.wait_for_frames()
            frameset = self.align.process(frames)
            depth_frame = frameset.get_depth_frame()
            color_frame = frameset.get_color_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            return color_image, depth_image