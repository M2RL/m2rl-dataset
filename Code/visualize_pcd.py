import cv2
import numpy as np
import open3d as o3d



class RGBDFeatureExtractor:
    def __init__(self, voxel_size=0.01):
        self.voxel_size = voxel_size

    def extract_features(self, rgb_image, depth_image, camera_intrinsics):
        # Convert RGB and depth images to Open3D formats
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_image), o3d.geometry.Image(depth_image), depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=False)

        # Create point cloud from RGBD image
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, o3d.camera.PinholeCameraIntrinsic(camera_intrinsics['width'], camera_intrinsics['height'], camera_intrinsics['fx'], camera_intrinsics['fy'], camera_intrinsics['cx'], camera_intrinsics['cy']))

        # Downsample point cloud using voxel grid filter
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # Estimate normals for downsampled point cloud
        pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Extract FPFH features
        fpfh_radius = 0.05
        fpfh_feature = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius * 2, max_nn=100))

        return pcd, pcd_down, fpfh_feature

# Example usage
if __name__ == "__main__":
    # Load RGB and depth images
    file_path = "<path-to-rgb>"
    rgb_image = cv2.imread(f"{file_path}/kinect1_color/27.png")
    depth_image = cv2.imread(f"{file_path}/kinect1_depth/27.png")

    # Camera intrinsics

    intrinsics = [[911.19207764,   0.        , 964.27746582],
       [  0.        , 910.90966797, 547.20727539],
       [  0.        ,   0.        ,   1.        ]]

    camera_intrinsics = {
        'width': 1280,
        'height': 720,
        'fx': 911.19,
        'fy': 910.91,
        'cx': 964.28,
        'cy': 547.20
    }

    # Create feature extractor
    extractor = RGBDFeatureExtractor(voxel_size=0.01)

    # Extract features
    pcd, pcd_down, fpfh_feature = extractor.extract_features(rgb_image, depth_image, camera_intrinsics)

    # Visualize downsampled point cloud
    o3d.visualization.draw_geometries([pcd])