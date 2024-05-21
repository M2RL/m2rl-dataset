import numpy as np


class Observation(object):
    """Storage for both visual and low-dimensional observations."""

    def __init__(self,                 
                 gripper_pose: np.ndarray,
                 gripper_rot: np.ndarray,
                 gripper_open: np.bool_,                 
                 joint_velocities: np.ndarray, 
                 controller_axis: np.ndarray,
                 controller_button: np.ndarray,
                 controller_hat: np.ndarray,
                 ignore_collisions: np.bool_ = np.zeros(1, dtype=np.bool_),
                 right_shoulder_rgb: np.ndarray = None,
                 left_shoulder_rgb: np.ndarray = None,
                 wrist_rgb: np.ndarray = None,
                 right_shoulder_depth: np.ndarray = None,
                 left_shoulder_depth: np.ndarray = None,
                 wrist_depth: np.ndarray = None,
                 ):
        
        self.gripper_pose = np.concatenate([gripper_pose, gripper_rot, gripper_open])
        self.gripper_rot = gripper_rot
        self.gripper_open = gripper_open
        self.ignore_collisions = ignore_collisions
        self.controller_axis = controller_axis
        self.controller_button = controller_button
        self.controller_hat = controller_hat
        self.joint_velocities = np.asarray(joint_velocities)
        self.right_shoulder_rgb = right_shoulder_rgb
        self.left_shoulder_rgb = left_shoulder_rgb
        self.wrist_rgb = wrist_rgb
        self.right_shoulder_depth = right_shoulder_depth
        self.left_shoulder_depth = left_shoulder_depth
        self.wrist_depth = wrist_depth
        
    def get_low_dim_data(self) -> np.ndarray:
        """Gets a 1D array of all the low-dimensional obseervations.

        :return: 1D array of observations.
        """
        print (self.controller_hat.shape, self.controller_button.shape, self.controller_axis.shape)
        print (self.gripper_pose.shape, self.gripper_rot.shape, self.gripper_open.shape)
        low_dim_data = [] if self.gripper_open is None else [[self.gripper_open]]
        for data in [self.gripper_pose, self.gripper_rot, self.gripper_open, 
                    self.controller_axis, self.controller_button, self.controller_hat]:
            if data is not None:
                low_dim_data.append(data)
        return np.concatenate(low_dim_data) if len(low_dim_data) > 0 else np.array([])
