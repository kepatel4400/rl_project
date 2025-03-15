import os
import numpy as np
import pybullet as p
import pybullet_data
import time
import cv2
from gymnasium import Env, spaces
import matplotlib.pyplot as plt
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s"  # Define the format of log messages
    # NOTE: To save logs to a file, uncomment the following line
    # handlers=[
    #     logging.FileHandler("app.log"),  # Save logs to a file
    #     logging.StreamHandler()  # Also print logs to console
    # ]
)

class RoboticPushingEnv(Env):
    """
    A PyBullet-based environment for training a robotic arm to push objects to a target area
    using multimodal observations (vision and proprioception).
    """
    
    def __init__(self, render_mode=None, max_steps=1000):
        """
        Initialize the robotic pushing environment.
        
        Args:
            render_mode (str, optional): Rendering mode ('human', 'rgb_array', or None)
            max_steps (int): Maximum number of steps per episode
        """
        super().__init__()
        
        # Environment parameters
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.step_counter = 0
        
        # Connect to PyBullet
        self.physics_client_id = self._connect_physics_client()
        
        # Set simulation parameters
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)  # 0: stepped mode, 1: real-time mode
        
        # Initialize environment elements
        self.plane_id = None
        self.table_id = None
        self.robot_id = None
        self.object_id = None
        self.target_id = None
        
        # Position parameters
        self.table_position = [0.5, 0, 0]
        self.robot_position = [0, 0, 0.6]
        self.object_start_position = [0.5, -0.3, 0.65]
        self.target_position = [0.6, 0.0, 0.625]
        
        # Camera parameters
        self.camera_position = [1.2, 0.0, 1.2]
        self.camera_target = [0.7, 0, 0.8]
        self.camera_up_vector = [0, 0, 1]
        self.camera_fov = 60
        self.camera_aspect = 1.0
        self.camera_near_val = 0.01
        self.camera_far_val = 2.0
        self.camera_width = 224
        self.camera_height = 224
        
        # Robot control parameters
        self.end_effector_index = 11  # End effector link index for the Panda robot
        self.num_joints = 0
        self.target_joint_positions = None
        
        # Create view and projection matrices for the camera
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_position,
            cameraTargetPosition=self.camera_target,
            cameraUpVector=self.camera_up_vector
        )
        
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=self.camera_aspect,
            nearVal=self.camera_near_val,
            farVal=self.camera_far_val
        )
        
        # Initialize the environment
        self._load_environment()
        
        # Define multimodal observation space
        # Visual: object bounding box (4 values: x, y, width, height)
        # Target: 2D position on table (x, y)
        # Proprioceptive: end-effector position (3) and velocity (3)
        self.observation_space = spaces.Dict({
            'vision': spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),  # Normalized bbox
            'target': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),  # Target x,y
            'proprioception': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)  # End-effector pos & vel
        })
        
        # Define action space (end-effector movement in x, y directions)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    
    def _connect_physics_client(self):
        """Connect to the PyBullet physics server."""
        if self.render_mode == "human":
            # return p.connect(p.GUI)
            # Suppress PyBullet GUI output
            with open(os.devnull, 'w') as devnull:
                original_stdout = os.dup(1)  # Duplicate stdout
                os.dup2(devnull.fileno(), 1)  # Redirect stdout to devnull
                client = p.connect(p.GUI)
                os.dup2(original_stdout, 1)  # Restore original stdout
            return client
        
        else:
            return p.connect(p.DIRECT)
    
    def _load_environment(self):
        """Load all objects in the environment."""
        # Reset simulation
        p.resetSimulation()
        
        # Load plane (ground)
        self.plane_id = p.loadURDF("plane.urdf", basePosition=[0, 0, 0])
        
        # Load table
        self.table_id = p.loadURDF("table/table.urdf", basePosition=self.table_position)
        
        # Load robot arm (Panda)
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", 
                                   basePosition=self.robot_position, 
                                   useFixedBase=True)
        
        # Get number of joints
        self.num_joints = p.getNumJoints(self.robot_id)
        
        # Set initial robot configuration
        self._reset_robot()
        
        # Load pushable object
        self.object_id = p.loadURDF("cube_small.urdf", basePosition=self.object_start_position)
        
        # Create target visual (green area)
        target_visual = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[0.05, 0.05, 0.005], 
            rgbaColor=[0, 1, 0, 0.6]
        )
        self.target_id = p.createMultiBody(
            baseVisualShapeIndex=target_visual, 
            basePosition=self.target_position
        )
        
        # Add camera visualization
        self._add_camera_visualization()
    
    def _reset_robot(self):
        """Reset robot to home position."""
        # Initial joint positions for home position
        initial_joint_positions = [0, 0.1, 0, -0.5, 0, 0.5, 0.7, 0.0, 0.0]
        
        # Set joint positions
        for i in range(self.num_joints):
            if i < len(initial_joint_positions):
                p.resetJointState(self.robot_id, i, initial_joint_positions[i])
            
            # Disable motors for now - we'll control positions directly
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=initial_joint_positions[i] if i < len(initial_joint_positions) else 0,
                force=500
            )
    
    def _add_camera_visualization(self):
        """Add visual elements to represent the camera in the simulation."""
        # Create a small red sphere to represent camera position
        camera_visual = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=0.03, 
            rgbaColor=[1, 0, 0, 0.8]
        )
        self.camera_body = p.createMultiBody(
            baseVisualShapeIndex=camera_visual, 
            basePosition=self.camera_position
        )
        
        # Create a line from camera to target
        p.addUserDebugLine(
            self.camera_position,
            self.camera_target,
            lineColorRGB=[1, 0, 0],
            lineWidth=2.0
        )
    
    def _get_camera_image(self):
        """Capture an image from the camera."""
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix
        )
        
        return rgb_img, seg_img
    
    def _get_object_bounding_box(self, rgb_img, seg_img):
        """
        Get the bounding box of the pushable object from segmentation image.
        
        Returns:
            bbox (np.ndarray): [x, y, width, height] normalized to [0, 1]
        """
        # TODO: check whether we can segment using the default seg_img
        # TODO: later train a nano YOLOv11 model to get bbox
        # Get object segmentation mask
        object_mask = (seg_img == self.object_id).astype(np.uint8)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box from the largest contour
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Normalize to [0, 1]
            x_norm = x / self.camera_width
            y_norm = y / self.camera_height
            w_norm = w / self.camera_width
            h_norm = h / self.camera_height
            
            return np.array([x_norm, y_norm, w_norm, h_norm], dtype=np.float32)
        
        # Return default bbox if object is not visible
        return np.zeros(4, dtype=np.float32)
    
    def _get_target_position_2d(self):
        """
        Get the 2D position of the target area in the table frame.
        
        Returns:
            np.ndarray: Normalized target [x, y] position
        """
        # TODO: can we get target position using seg_img?
        # TODO: train a nano YOLOv11 model to get bbox
        # Get the target position relative to the table center
        target_x = self.target_position[0] - self.table_position[0]
        target_y = self.target_position[1] - self.table_position[1]
        
        # Normalize to [-1, 1] (assuming table is 1.0 x 1.0 m)
        target_x_norm = target_x * 2.0
        target_y_norm = target_y * 2.0
        
        return np.array([target_x_norm, target_y_norm], dtype=np.float32)
    
    def _get_proprioceptive_data(self):
        """
        Get proprioceptive data: end-effector position and velocity.
        
        Returns:
            np.ndarray: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
        """
        # Get end-effector state
        end_effector_state = p.getLinkState(self.robot_id, self.end_effector_index, computeLinkVelocity=1)
        
        # Extract position and velocity
        position = end_effector_state[0]  # Position in world frame
        velocity = end_effector_state[6]  # Linear velocity in world frame
        
        return np.array([*position, *velocity], dtype=np.float32)
    
    def _get_observation(self):
        """
        Get the multimodal observation from the environment.
        
        Returns:
            dict: Dictionary with vision, target, and proprioception data
        """
        # Get camera image
        rgb_img, depth_img, seg_img = self._get_camera_image()
        
        # Extract observations
        object_bbox = self._get_object_bounding_box(rgb_img, seg_img)
        target_pos = self._get_target_position_2d()
        proprio_data = self._get_proprioceptive_data()
        
        return {
            'vision': object_bbox,
            'target': target_pos,
            'proprioception': proprio_data
        }
    
    def _is_object_at_target(self):
        """
        Check if the pushable object is at the target position.
        
        Returns:
            bool: True if object is at target, False otherwise
        """
        # Get object position
        object_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        
        # Calculate distance to target in the xy-plane
        dist = np.sqrt((object_pos[0] - self.target_position[0])**2 + 
                      (object_pos[1] - self.target_position[1])**2)
        
        # Check if distance is within threshold
        return dist < 0.05  # 5 cm threshold
    
    def _calculate_reward(self):
        """
        Calculate reward based on the current state.
        
        Returns:
            float: Reward value
        """
        # Get object and target positions
        object_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        
        # Calculate distance to target in the xy-plane
        dist = np.sqrt((object_pos[0] - self.target_position[0])**2 + 
                      (object_pos[1] - self.target_position[1])**2)
        
        # Proximity reward (inverse of distance)
        proximity_reward = 1.0 / (1.0 + 10.0 * dist)
        
        # Goal completion reward
        goal_reward = 10.0 if self._is_object_at_target() else 0.0
        
        # Time step penalty
        time_penalty = -0.01
        
        # Total reward
        reward = proximity_reward + goal_reward + time_penalty
        
        return reward
    
    def _apply_action(self, action):
        """
        Apply action to control the robot arm.
        
        Args:
            action (np.ndarray): [delta_x, delta_y] for end-effector movement
        """
        # Scale action to appropriate range (e.g., cm)
        action_scale = 0.02  # 2 cm per step when action = 1.0
        delta_x = action[0] * action_scale
        delta_y = action[1] * action_scale
        
        # Get current end-effector position
        end_effector_state = p.getLinkState(self.robot_id, self.end_effector_index)
        current_position = end_effector_state[0]
        
        # Calculate target position (only modify x, y)
        target_position = [
            current_position[0] + delta_x,
            current_position[1] + delta_y,
            current_position[2]  # Keep z-coordinate the same
        ]
        
        # Use inverse kinematics to get joint positions for target end-effector position
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            target_position
        )
        
        # Apply joint positions (only to the first 7 joints of the Panda arm)
        for i in range(min(7, len(joint_positions))):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=500
            )
        
        # Step the simulation multiple times to let the arm move
        for _ in range(10):
            p.stepSimulation()
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        
        Returns:
            observation (dict): Initial observation
            info (dict): Additional information
        """
        super().reset(seed=seed)
        
        # Reset step counter
        self.step_counter = 0
        
        # Reset simulation
        self._load_environment()
        
        # Randomize object position (optional)
        if options and 'randomize' in options and options['randomize']:
            x = np.random.uniform(0.3, 0.7)
            y = np.random.uniform(-0.3, 0.3)
            self.object_start_position = [x, y, 0.65]
            p.resetBasePositionAndOrientation(
                self.object_id,
                self.object_start_position,
                [0, 0, 0, 1]
            )
        
        # Get initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (np.ndarray): [delta_x, delta_y] for end-effector movement
        
        Returns:
            observation (dict): Current observation
            reward (float): Reward for this step
            terminated (bool): Whether the episode is terminated
            truncated (bool): Whether the episode is truncated (e.g., due to max_steps)
            info (dict): Additional information
        """
        # Apply action
        self._apply_action(action)
        
        # Increment step counter
        self.step_counter += 1
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self._is_object_at_target()
        truncated = self.step_counter >= self.max_steps
        
        # Additional info
        info = {
            'is_success': terminated,
            'step_count': self.step_counter
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            np.ndarray or None: RGB image if render_mode is 'rgb_array', None otherwise
        """
        if self.render_mode == "rgb_array":
            rgb_img, _, _ = self._get_camera_image()
            return rgb_img
        
        return None
    
    def close(self):
        """Close the environment and disconnect from PyBullet."""
        p.disconnect(self.physics_client_id)


# Example usage
if __name__ == "__main__":
    # Create environment
    env = RoboticPushingEnv(render_mode="human", max_steps=100000)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Run a few random steps
    for _ in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            logging.debug(f"Episode terminated with reward: {reward}")
            obs, _ = env.reset()
    
    logging.info("Simulation complete!")
    print(f"{'*'*20}*")
    # Close environment
    env.close()