import os
import numpy as np
import pybullet as p
import pybullet_data
import time
import cv2
from gymnasium import Env, spaces
import matplotlib.pyplot as plt
import logging
from urdf_models import models_data

'''
🐞 (Ladybug): For debugging.
ℹ️ (Information): For informational messages.
🚨 (Alert/Siren): For errors or warnings.
✅ (Checkmark): For completion or success.
⏱️ (Timer Clock): For timing or performance metrics.
🟡 (Warning)
'''

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
        # p.setGravity(0, 0, -9.81)
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
        self.object_start_position = [0.5, -0.1, 0.65]
        self.target_position = [0.6, 0.0, 0.625]

        self.table_width = 1.0
        self.table_height = 0.6
        self.table_length = 1.5
        
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
        self.end_effector_index = 11  # Using 'panda_grasptarget_hand' as the pushing point
        self.num_joints = 0
        self.joint_limits = [
            (-2.9671, 2.9671),  # panda_joint1
            (-1.8326, 1.8326),  # panda_joint2
            (-2.9671, 2.9671),  # panda_joint3
            (-3.1416, 0.0),     # panda_joint4
            (-2.9671, 2.9671),  # panda_joint5
            (-0.0873, 3.8223),  # panda_joint6
            (-2.9671, 2.9671),  # panda_joint7
        ]
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

        # Load URDF models
        self.models = models_data.model_lib()
        self.ycb_object = "pudding_box" # Available objects: ['cracker_box', 'sugar_box', "soap", "pudding_box", "gelatin_box", "blue_tea_box"]
        
        # Initialize the environment
        self._load_environment()

        # Calculate and store the target position 2D
        _, seg_img = self._get_camera_image()
        self.target_position_2d = self._get_target_position_2d(seg_img)
        
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

        # User parameters
        self.distance_threshold = 0.05  # Distance threshold for object-target proximity    
    
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
        # self.object_id = p.loadURDF("cube_small.urdf", basePosition=self.object_start_position)
        # Load a YCB object
        available_objects = ['cracker_box', 'sugar_box', "soap", "pudding_box", "gelatin_box", "blue_tea_box"]
        flags = p.URDF_USE_INERTIA_FROM_FILE
        self.object_id = p.loadURDF(self.models[self.ycb_object], basePosition=self.object_start_position, flags=flags)
        
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

        p.setGravity(0, 0, -9.81)
        
        # Add camera visualization
        self._add_camera_visualization()
    
    def _reset_robot(self):
        """Reset robot to home position."""
        # Initial joint positions for home position - configured for pushing task
        # These values position the end effector above the table at a good height for pushing
        # initial_joint_positions = [0, -0.3, 0, -2.2, 0, 2.0, 0.8]
        initial_joint_positions = [0.0, 0.347, -0.0, -2.431, -0.0, 2.732, 0.625]
        
        # Get current number of joints
        self.num_joints = p.getNumJoints(self.robot_id)
        
        # Set joint positions for the 7 controllable joints
        for i in range(7):
            p.resetJointState(self.robot_id, i, initial_joint_positions[i])
            
            # Set position control with appropriate force limits from joint info
            max_force = 87.0 if i < 4 else 12.0  # First 4 joints have higher force limits
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=initial_joint_positions[i],
                force=max_force
            )
        
        # Set gripper finger joints to close position
        p.resetJointState(self.robot_id, 9, 0.0)  # Left finger closed 
        p.resetJointState(self.robot_id, 10, 0.0)  # Right finger closed
        
        # Control gripper
        for i in [9, 10]:  # Finger joints
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0, # Fingers closed
                force=20.0
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
        # p.addUserDebugLine(
        #     self.camera_position,
        #     self.camera_target,
        #     lineColorRGB=[1, 0, 0],
        #     lineWidth=2.0
        # )
    
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
        # We need to convert segmentation image to a format we can process
        # In PyBullet, segmentation images might need to be processed differently
        # based on how the ids are stored
        
        # Convert segmentation image from rgba to single channel
        seg_img_single = seg_img & ((1 << 24) - 1)  # Extract object ID from the seg_img
        
        # Create mask where the object ID matches
        object_mask = np.zeros_like(seg_img_single, dtype=np.uint8)
        object_mask[seg_img_single == self.object_id] = 255
        
        # Debug visualization if needed
        cv2.imwrite('debug_mask.png', object_mask)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours and len(contours) > 0:
            # Get bounding box from the largest contour
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Normalize to [0, 1]
            x_norm = x / self.camera_width
            y_norm = y / self.camera_height
            w_norm = w / self.camera_width
            h_norm = h / self.camera_height
            
            # logging.info(f"✅ [vision] Found object with bbox")
            return np.array([x_norm, y_norm, w_norm, h_norm], dtype=np.float32)
        
        # If object not found in segmentation, try color-based detection as fallback
        # This might be useful if segmentation doesn't work as expected
        if np.sum(object_mask) == 0:
            # Convert RGB to HSV for better color segmentation
            hsv = cv2.cvtColor(rgb_img[:,:,:3].astype(np.uint8), cv2.COLOR_RGB2HSV)
            
            # Define range for red color (assuming cube is red)
            # This would need to be adjusted based on actual object color
            lower_red = np.array([0, 120, 70])
            upper_red = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            
            lower_red = np.array([170, 120, 70])
            upper_red = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red, upper_red)
            
            color_mask = mask1 + mask2
            
            # Find contours in color mask
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours and len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(contour)
                
                # Normalize to [0, 1]
                x_norm = x / self.camera_width
                y_norm = y / self.camera_height
                w_norm = w / self.camera_width
                h_norm = h / self.camera_height
                
                logging.info(f"🟡 [vision] Found object with color-based detection")

                return np.array([x_norm, y_norm, w_norm, h_norm], dtype=np.float32)
        
        logging.info(f"❌ [vision] Object not found")
        # Return default bbox if object is not visible
        return np.array([0.5, 0.5, 0.1, 0.1], dtype=np.float32)
    
    def _get_target_position_2d(self, seg_img):
        """
        Get the 2D position of the target area in the table frame using the segmentation image.
        
        Args:
            seg_img (np.ndarray): Segmentation image from the camera
        
        Returns:
            np.ndarray: Normalized target [x, y] position
        """
        # Convert segmentation image from rgba to single channel
        seg_img_single = seg_img & ((1 << 24) - 1)  # Extract object ID from the seg_img
        
        # Create mask where the target ID matches
        target_mask = np.zeros_like(seg_img_single, dtype=np.uint8)
        target_mask[seg_img_single == self.target_id] = 255

        # Debug visualization if needed
        cv2.imwrite('debug_target_mask.png', target_mask)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours and len(contours) > 0:
            # Get bounding box from the largest contour
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate the center of the bounding box
            target_x = x + w / 2
            target_y = y + h / 2
            
            # Normalize to [-1, 1] (assuming table is 1.0 x 1.0 m)
            target_x_norm = (target_x / self.camera_width) * 2.0 - 1.0
            target_y_norm = (target_y / self.camera_height) * 2.0 - 1.0
            
            return np.array([target_x_norm, target_y_norm], dtype=np.float32)
        
        # If target not found, return default position
        return np.array([0.0, 0.0], dtype=np.float32)
    
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
        rgb_img, seg_img = self._get_camera_image()
        
        # Extract observations
        object_bbox = self._get_object_bounding_box(rgb_img, seg_img)
        proprio_data = self._get_proprioceptive_data()
        
        return {
            'vision': object_bbox,
            'target': self.target_position,
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
        return dist < self.distance_threshold

    def _calculate_reward(self):
        """
        Reward engineering is a crucial part of RL algorithm design.
        Calculate reward based on the current state. 
        In total, there are 6 rewards:
        1. Time step penalty (dense reward)
        2. Goal completion reward (sparse reward)
        3. Proximity reward (dense reward)
        4. Progress reward (dense reward)
        5. Approach reward (dense reward)
        6. Movement reward (dense reward)

        # TODO: look into below suggestions
        During testing and training, you might consider these adjustments if needed:
        - Increase goal_reward_weight if the agent isn't sufficiently motivated to reach the target
        - Decrease time_penalty_weight if the agent seems too rushed and makes mistakes
        - Increase approach_reward_weight if the agent has trouble finding the object
        
        Returns:
            float: Reward value
        """

        # Reward weights for easier tuning
        time_penalty_weight = 1.0
        goal_reward_weight = 1.0
        distance_reward_weight = 1.0
        progress_reward_weight = 2.0
        approach_reward_weight = 1.0
        movement_reward_weight = 2.0

        # Reward 1: Time step penalty (encourages efficiency) (dense reward)
        time_penalty = -0.01
        
        # Reward 2: Goal completion reward (sparse reward)
        goal_reward = 10.0 if self._is_object_at_target() else 0.0

        # Get object and target positions
        object_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        
        # Calculate distance to target in the xy-plane
        dist = np.sqrt((object_pos[0] - self.target_position[0])**2 + 
                      (object_pos[1] - self.target_position[1])**2)
        
        # Reward 3: Proximity reward (dense reward)
        # Distance-based reward (bounded to avoid excessive penalties)
        # Using exponential function to smooth out the penalty
        distance_reward = -1.0 * (1.0 - np.exp(-2.0 * dist))
        
        # Reward 4: Progress reward (dense reward)
        # Progress reward (for reducing distance)
        if hasattr(self, 'prev_dist'):
            # Only reward positive progress (decreasing distance)
            progress_reward = max(0, self.prev_dist - dist) * 2.0
            self.prev_dist = dist
        else:
            progress_reward = 0
            self.prev_dist = dist
        
        # Reward 5: End-effector to object proximity reward (dense reward)
        # End-effector to object proximity reward (encourages interaction)
        ee_pos = p.getLinkState(self.robot_id, self.end_effector_index)[0]
        ee_to_object_dist = np.sqrt((ee_pos[0] - object_pos[0])**2 + 
                                    (ee_pos[1] - object_pos[1])**2)
        approach_reward = 0.2 / (1.0 + 10.0 * ee_to_object_dist)
        
        # Reward 6: Object movement reward (dense reward)
        # Object movement reward (encourage pushing rather than just contact)
        if hasattr(self, 'prev_object_pos'):
            object_movement = np.sqrt((object_pos[0] - self.prev_object_pos[0])**2 + 
                                     (object_pos[1] - self.prev_object_pos[1])**2)
            movement_reward = min(0.5, object_movement * 5.0)  # Cap at 0.5 to prevent erratic behavior
        else:
            movement_reward = 0
        self.prev_object_pos = object_pos
        
        # Total reward
        # reward = time_penalty + goal_reward + distance_reward + progress_reward + approach_reward + movement_reward
        reward = (time_penalty_weight * time_penalty +
                  goal_reward_weight * goal_reward +
                  distance_reward_weight * distance_reward +
                  progress_reward_weight * progress_reward +
                  approach_reward_weight * approach_reward +
                  movement_reward_weight * movement_reward)
        
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
        current_orientation = end_effector_state[1]
        
        # Calculate target position (only modify x, y)
        target_position = [
            current_position[0] + delta_x,
            current_position[1] + delta_y,
            current_position[2]  # Keep z-coordinate the same
        ]

        # Safety check
        if not self._is_pose_safe(target_position, current_orientation):
            # Try smaller movement if original is unsafe
            scale_factor = 0.5
            safe_target = None
            
            # Try progressively smaller actions until finding a safe one
            while scale_factor >= 0.1:
                adjusted_position = [
                    current_position[0] + (delta_x * scale_factor),
                    current_position[1] + (delta_y * scale_factor),
                    current_position[2]
                ]
                
                if self._is_pose_safe(adjusted_position, current_orientation):
                    safe_target = adjusted_position
                    logging.info(f"🟡 [safety] Reduced action to {scale_factor*100}% for safety")
                    break
                    
                scale_factor *= 0.5
            
            if safe_target is None:
                logging.info("🚨 [safety] No safe action found, skipping")
                return
            
            target_position = safe_target
        
        # Use inverse kinematics to get joint positions for target end-effector position
        # We maintain the current orientation for stable pushing
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            target_position,
            current_orientation,
            maxNumIterations=100,
            residualThreshold=1e-4
        )
        
        # Apply joint positions (only to the first 7 joints of the Panda arm)
        for i in range(7):
            # Get appropriate force limits
            max_force = 87.0 if i < 4 else 12.0  # Based on the joint info
            
            # Clip to joint limits
            lower_limit, upper_limit = self.joint_limits[i]
            target_position = min(max(joint_positions[i], lower_limit), upper_limit)
            
            # Set position control
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position,
                force=max_force,
                maxVelocity=2.0  # Limit velocity for smoother movement
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

        # Visualize safe workspace for debugging
        if self.render_mode == "human":
            self._visualize_safe_workspace()
        
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
    
    def _is_pose_safe(self, target_position, target_orientation=None):
        """
        Check if a given end-effector pose is safe (no collisions with table or self).
        
        Args:
            target_position (list): Target position [x, y, z]
            target_orientation (list, optional): Target orientation as quaternion
            
        Returns:
            bool: True if the pose is safe, False otherwise
        """
        # Get current end-effector orientation if not provided
        if target_orientation is None:
            current_orientation = p.getLinkState(self.robot_id, self.end_effector_index)[1]
            target_orientation = current_orientation
        
        # 1. Check table height bounds
        min_safe_height = self.table_height + 0.02  # Minimum safe height above table
        
        if target_position[2] < min_safe_height:
            logging.info("🚨 [safety] Target position too low, would collide with table")
            return False
        
        # 2. Check table bounds (assuming the table is rectangular)
        table_half_length = self.table_length/2  # Adjust based on your table dimensions
        table_half_width = self.table_width/2   # Adjust based on your table dimensions
        
        table_min_x = self.table_position[0] - table_half_length
        table_max_x = self.table_position[0] + table_half_length
        table_min_y = self.table_position[1] - table_half_width
        table_max_y = self.table_position[1] + table_half_width
        
        # Check if target is far outside table bounds (too far is also unsafe)
        max_reach = 0.9  # Maximum safe reach of the robot arm

        print(f"[DENUGGGGGGGGGGGGGGGGG]")
        print(f"table_min_x: {table_min_x}")
        print(f"table_max_x: {table_max_x}")
        print(f"table_min_y: {table_min_y}")
        print(f"table_max_y: {table_max_y}")
        print(f"target_position: {target_position}")
        print(f"max_reach: {max_reach}")
        print(f" 1st condition: {target_position[0] < table_min_x - 0.1}")
        print(f" 2nd condition: {target_position[0] > table_max_x + 0.1}")
        print(f" 3rd condition: {target_position[1] < table_min_y - 0.1}")
        print(f" 4th condition: {target_position[1] > table_max_y + 0.1}")
        print(f"np linalg norm: {np.linalg.norm(np.array(target_position))}")
        print(f"[DENUGGGGGGGGGGGGGGGGG]")

        if (target_position[0] < table_min_x - 0.1 or 
            target_position[0] > table_max_x + 0.1 or
            target_position[1] < table_min_y - 0.1 or
            target_position[1] > table_max_y + 0.1 or
            np.linalg.norm(np.array(target_position)) > max_reach):
            logging.info("🚨 [safety] Target position outside safe workspace")
            return False
        
        # 3. Check if IK solution exists and is feasible
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            target_position,
            target_orientation,
            maxNumIterations=100,
            residualThreshold=1e-4
        )
        
        # Save current joint positions
        current_joints = []
        for i in range(7):  # 7 DOF for Panda arm
            current_joints.append(p.getJointState(self.robot_id, i)[0])
        
        # Check if IK solution is within joint limits
        for i in range(7):
            lower_limit, upper_limit = self.joint_limits[i]
            if joint_positions[i] < lower_limit or joint_positions[i] > upper_limit:
                logging.info(f"🚨 [safety] Joint {i} would exceed limits")
                return False
        
        # 4. Use PyBullet's built-in collision detection
        # Temporarily move to target position
        for i in range(7):
            p.resetJointState(self.robot_id, i, joint_positions[i])
        
        # Check for collisions
        collision = False
        if p.getContactPoints(bodyA=self.robot_id, bodyB=self.table_id):
            logging.info("🚨 [safety] Would collide with table at target position")
            collision = True
        
        # Also check for self-collisions
        for i in range(p.getNumJoints(self.robot_id)):
            for j in range(i+1, p.getNumJoints(self.robot_id)):
                if p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id, linkIndexA=i, linkIndexB=j):
                    logging.info(f"🚨 [safety] Self-collision detected between links {i} and {j}")
                    collision = True
                    break
            if collision:
                break
        
        # Restore original joint positions
        for i in range(7):
            p.resetJointState(self.robot_id, i, current_joints[i])
        
        return not collision
    
    def _visualize_safe_workspace(self):
        """
        Visualize the safe workspace boundaries for debugging.
        """
        if self.render_mode != "human":
            return
        
        # Table dimensions
        table_half_length = 0.75
        table_half_width = 0.5
        
        # Table corners in world coordinates
        table_min_x = self.table_position[0] - table_half_length
        table_max_x = self.table_position[0] + table_half_length
        table_min_y = self.table_position[1] - table_half_width
        table_max_y = self.table_position[1] + table_half_width
        
        # Safe height above table
        safe_height = self.table_height + 0.02
        
        # Draw table outline at safe height (using debug lines)
        # Create a rectangle outline
        p.addUserDebugLine([table_min_x, table_min_y, safe_height], 
                        [table_max_x, table_min_y, safe_height], 
                        [0, 1, 0], 2.0)  # Green line
        
        p.addUserDebugLine([table_max_x, table_min_y, safe_height], 
                        [table_max_x, table_max_y, safe_height], 
                        [0, 1, 0], 2.0)
        
        p.addUserDebugLine([table_max_x, table_max_y, safe_height], 
                        [table_min_x, table_max_y, safe_height], 
                        [0, 1, 0], 2.0)
        
        p.addUserDebugLine([table_min_x, table_max_y, safe_height], 
                        [table_min_x, table_min_y, safe_height], 
                        [0, 1, 0], 2.0)
        
        # Show safe workspace limits as a transparent box
        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[table_half_length, table_half_width, 0.001],
            rgbaColor=[0, 1, 0, 0.1]  # Semi-transparent green
        )
        
        p.createMultiBody(
            baseVisualShapeIndex=visual_id,
            basePosition=[self.table_position[0], self.table_position[1], safe_height]
        )
        
        # Text annotation
        p.addUserDebugText(
            "Safe Workspace",
            [self.table_position[0], self.table_position[1], safe_height + 0.05],
            [0, 0.5, 0],
            textSize=1.5
        )

def test_environment_2(test_safety=True, show_plots=True, max_steps=300):
    """
    Comprehensive test function for the RoboticPushingEnv.
    
    Args:
        test_safety (bool): Whether to include safety feature tests
        show_plots (bool): Whether to show plots during/after testing
        max_steps (int): Maximum steps for the environment
    """
    print("\n" + "="*50)
    print("STARTING ENVIRONMENT TEST")
    print("="*50)
    
    # Create and reset environment
    env = RoboticPushingEnv(render_mode="human", max_steps=max_steps)
    obs, _ = env.reset()
    
    # Visualize the safe workspace
    env._visualize_safe_workspace()
    
    # Initialize tracking variables
    rewards = []
    ee_positions = []
    object_positions = []
    safety_violations = 0
    step_count = 0
    
    # Get initial object position for tracking
    object_pos, _ = p.getBasePositionAndOrientation(env.object_id)
    object_positions.append((object_pos[0], object_pos[1]))
    
    # Get initial end-effector position
    ee_pos = p.getLinkState(env.robot_id, env.end_effector_index)[0]
    ee_positions.append((ee_pos[0], ee_pos[1]))
    
    print("\nPart 1: Basic Movement Test")
    print("-"*30)
    
    # Standard action sequence - basic testing
    basic_actions = [
        [0.5, 0],    # Move right
        [0, 0.5],    # Move forward
        [-0.5, 0],   # Move left
        [0, -0.5]    # Move back
    ]
    
    for i, action in enumerate(basic_actions * 3):  # Repeat 3 times
        obs, reward, terminated, truncated, info = env.step(np.array(action))
        
        # Track data
        rewards.append(reward)
        ee_pos = p.getLinkState(env.robot_id, env.end_effector_index)[0]
        ee_positions.append((ee_pos[0], ee_pos[1]))
        object_pos, _ = p.getBasePositionAndOrientation(env.object_id)
        object_positions.append((object_pos[0], object_pos[1]))
        step_count += 1
        
        print(f"Step {step_count}, Action: {action}, Reward: {reward:.4f}")
        
        if terminated:
            print("✅ Goal reached during basic movement test!")
            break
        
        # Add small delay for visualization
        time.sleep(0.05)
    
    # Part 2: Safety feature testing (optional)
    if test_safety:
        print("\nPart 2: Safety Feature Test")
        print("-"*30)
        
        # Test extreme actions
        safety_test_actions = [
            [5.0, 0.0],     # Very large right movement
            [0.0, 5.0],     # Very large forward movement
            [-5.0, 0.0],    # Very large left movement
            [0.0, -5.0],    # Very large backward movement
            [3.0, 3.0],     # Large diagonal movement
            [-3.0, -3.0]    # Large opposite diagonal movement
        ]
        
        for i, action in enumerate(safety_test_actions):
            # Store position before applying action
            ee_pos_before = p.getLinkState(env.robot_id, env.end_effector_index)[0]
            
            # Get current end-effector position and test if action would be safe
            current_position = p.getLinkState(env.robot_id, env.end_effector_index)[0]
            current_orientation = p.getLinkState(env.robot_id, env.end_effector_index)[1]
            
            # Calculate target position
            action_scale = 0.02  # Same as in your _apply_action method
            delta_x = action[0] * action_scale
            delta_y = action[1] * action_scale
            target_position = [
                current_position[0] + delta_x,
                current_position[1] + delta_y,
                current_position[2]
            ]
            
            # Check if pose would be safe before action
            is_safe = env._is_pose_safe(target_position, current_orientation)
            print(f"Safety test {i+1}: Action {action} - {'✅ Safe' if is_safe else '❌ Unsafe'}")
            
            # Apply action
            obs, reward, terminated, truncated, info = env.step(np.array(action))
            
            # Track data
            rewards.append(reward)
            ee_pos_after = p.getLinkState(env.robot_id, env.end_effector_index)[0]
            ee_positions.append((ee_pos_after[0], ee_pos_after[1]))
            object_pos, _ = p.getBasePositionAndOrientation(env.object_id)
            object_positions.append((object_pos[0], object_pos[1]))
            step_count += 1
            
            # Calculate if the action was modified/limited
            dist_moved = np.sqrt((ee_pos_after[0] - ee_pos_before[0])**2 + 
                               (ee_pos_after[1] - ee_pos_before[1])**2)
            expected_dist = np.sqrt((delta_x)**2 + (delta_y)**2)
            
            # Check if movement was limited
            if dist_moved < expected_dist * 0.9:  # Allow for small numerical differences
                print(f"   Action was limited: moved {dist_moved:.4f}m vs expected {expected_dist:.4f}m")
                safety_violations += 1
            
            # Add delay for visualization
            time.sleep(0.2)
        
    # Part 3: Object pushing test - try to push toward the target
    print("\nPart 3: Object Pushing Test")
    print("-"*30)
    
    # Find direction from object to target
    object_pos, _ = p.getBasePositionAndOrientation(env.object_id)
    target_vec = [
        env.target_position[0] - object_pos[0],
        env.target_position[1] - object_pos[1]
    ]
    
    # Normalize
    target_dist = np.sqrt(target_vec[0]**2 + target_vec[1]**2)
    if target_dist > 0:
        target_vec = [target_vec[0]/target_dist, target_vec[1]/target_dist]
    
    # End-effector to object vector
    ee_pos = p.getLinkState(env.robot_id, env.end_effector_index)[0]
    to_object_vec = [
        object_pos[0] - ee_pos[0],
        object_pos[1] - ee_pos[1]
    ]
    
    # First move toward object
    print("Moving end-effector toward object...")
    to_object_dist = np.sqrt(to_object_vec[0]**2 + to_object_vec[1]**2)
    if to_object_dist > 0:
        to_object_vec = [to_object_vec[0]/to_object_dist, to_object_vec[1]/to_object_dist]
        
        for _ in range(10):
            action = [to_object_vec[0], to_object_vec[1]]
            obs, reward, terminated, truncated, info = env.step(np.array(action))
            
            # Update tracking
            rewards.append(reward)
            ee_pos = p.getLinkState(env.robot_id, env.end_effector_index)[0]
            ee_positions.append((ee_pos[0], ee_pos[1]))
            object_pos, _ = p.getBasePositionAndOrientation(env.object_id)
            object_positions.append((object_pos[0], object_pos[1]))
            step_count += 1
            
            # Check if we reached the object
            ee_to_obj_dist = np.sqrt((ee_pos[0] - object_pos[0])**2 + (ee_pos[1] - object_pos[1])**2)
            if ee_to_obj_dist < 0.05:
                print("   Reached object vicinity")
                break
                
            time.sleep(0.05)
    
    # Then try to push object toward target
    print("Pushing object toward target...")
    for _ in range(30):
        # Get updated positions
        object_pos, _ = p.getBasePositionAndOrientation(env.object_id)
        target_vec = [
            env.target_position[0] - object_pos[0],
            env.target_position[1] - object_pos[1]
        ]
        target_dist = np.sqrt(target_vec[0]**2 + target_vec[1]**2)
        
        if target_dist < env.distance_threshold:
            print("   ✅ Successfully pushed object to target!")
            break
            
        if target_dist > 0:
            target_vec = [target_vec[0]/target_dist, target_vec[1]/target_dist]
        
        # Use target direction as action
        action = [target_vec[0], target_vec[1]]
        obs, reward, terminated, truncated, info = env.step(np.array(action))
        
        # Update tracking
        rewards.append(reward)
        ee_pos = p.getLinkState(env.robot_id, env.end_effector_index)[0]
        ee_positions.append((ee_pos[0], ee_pos[1]))
        object_pos, _ = p.getBasePositionAndOrientation(env.object_id)
        object_positions.append((object_pos[0], object_pos[1]))
        step_count += 1
        
        if terminated:
            print("   ✅ Target reached! Success!")
            break
            
        time.sleep(0.05)
    
    # Generate test report
    print("\nTest Results Summary")
    print("-"*30)
    print(f"Total steps executed: {step_count}")
    print(f"Final reward: {rewards[-1]:.4f}")
    print(f"Safety violations detected: {safety_violations}")
    
    # Final object distance to target
    object_pos, _ = p.getBasePositionAndOrientation(env.object_id)
    final_dist = np.sqrt((object_pos[0] - env.target_position[0])**2 + 
                       (object_pos[1] - env.target_position[1])**2)
    print(f"Final object distance to target: {final_dist:.4f}m")
    print(f"Success: {'Yes' if final_dist < env.distance_threshold else 'No'}")
    
    # Plot results if requested
    if show_plots:
        # Trajectories plot
        plt.figure(figsize=(10, 8))
        
        # Plot table boundaries
        table_half_length = 0.4
        table_half_width = 0.4
        table_x = env.table_position[0]
        table_y = env.table_position[1]
        
        table_xs = [table_x-table_half_length, table_x+table_half_length, 
                  table_x+table_half_length, table_x-table_half_length, table_x-table_half_length]
        table_ys = [table_y-table_half_width, table_y-table_half_width, 
                  table_y+table_half_width, table_y+table_half_width, table_y-table_half_width]
        
        plt.plot(table_xs, table_ys, 'k--', label='Table Boundary')
        
        # Plot end-effector trajectory
        ee_xs, ee_ys = zip(*ee_positions)
        plt.plot(ee_xs, ee_ys, 'b-', label='End-Effector Path', linewidth=2)
        plt.plot(ee_xs[0], ee_ys[0], 'bo', markersize=8, label='Start')
        plt.plot(ee_xs[-1], ee_ys[-1], 'bx', markersize=8, label='End')
        
        # Plot object trajectory
        obj_xs, obj_ys = zip(*object_positions)
        plt.plot(obj_xs, obj_ys, 'r-', label='Object Path', linewidth=2)
        plt.plot(obj_xs[0], obj_ys[0], 'ro', markersize=8)
        plt.plot(obj_xs[-1], obj_ys[-1], 'rx', markersize=8)
        
        # Plot target
        plt.plot(env.target_position[0], env.target_position[1], 'g*', 
               markersize=12, label='Target')
        
        # Add threshold circle around target
        circle = plt.Circle((env.target_position[0], env.target_position[1]), 
                          env.distance_threshold, color='g', fill=False, 
                          linestyle='--', label='Success Threshold')
        plt.gca().add_patch(circle)
        
        plt.grid(True)
        plt.axis('equal')
        plt.title('Trajectories')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.legend()
        
        # Reward plot
        plt.figure(figsize=(10, 4))
        plt.plot(rewards, 'g-')
        plt.grid(True)
        plt.title('Rewards over Steps')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        
        plt.tight_layout()
        plt.show()
    
    print("\nTest completed. Closing environment...")
    env.close()
    return {
        'steps': step_count,
        'final_reward': rewards[-1] if rewards else 0,
        'final_distance': final_dist,
        'success': final_dist < env.distance_threshold,
        'safety_violations': safety_violations
    }

# Example usage
if __name__ == "__main__":

    # test_environment()
    test_results = test_environment_2(test_safety=True, show_plots=True, max_steps=300)
    print(f"\nFinal Results: {test_results}")

    '''
    # Create environment
    env = RoboticPushingEnv(render_mode="human", max_steps=100000)
    
    # Reset environment
    logging.info("ℹ️ Resetting environment...")
    obs, _ = env.reset()
    
    # Run a few random steps
    for _ in range(1000):
        action = env.action_space.sample()  # Random action

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            logging.debug(f"Episode terminated with reward: {reward}")
            obs, _ = env.reset()
    
    print(f"\n{'*'*20}*")
    logging.info("✅ Simulation complete!")
    print(f"{'*'*80}*")
    print(f"{'*'*80}*\n")
    # Close environment
    env.close()'
    '''