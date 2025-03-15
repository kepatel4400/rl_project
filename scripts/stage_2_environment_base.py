import os
import pybullet as p
import pybullet_data
import time
import numpy as np
import cv2

os.environ['PYBULLET_DEBUG'] = '0'


with open(os.devnull, 'w') as devnull:
    original_stdout = os.dup(1)  # Duplicate stdout
    os.dup2(devnull.fileno(), 1)  # Redirect stdout to devnull
    client = p.connect(p.GUI)
    os.dup2(original_stdout, 1)  # Restore original stdout

# Connect to PyBullet in GUI mode
# physicsClient = p.connect(p.GUI)

p.resetSimulation()

# Set simulation parameters
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

p.setRealTimeSimulation(0) #0 or 1 // 0 (stepped mode): simulation runs in steps, 1 (real-time mode): simulation runs continuously in real-time

# Load the plane (ground)
p.loadURDF("plane.urdf", basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1])

# Adjusted Table position to be above the plane
TABLE_POSITION = [0.5, 0, 0]
table = p.loadURDF("table/table.urdf", basePosition=TABLE_POSITION)

# Adjusted robotic arm (Panda) height to match table height
ROBOT_POSITION = [0, 0, 0.6]
panda = p.loadURDF("franka_panda/panda.urdf", basePosition=ROBOT_POSITION, useFixedBase=True)

# [DEBUG only]
# To get total number of joints and joint info
# print("[DEBUG] Total Joints: ", p.getNumJoints(panda))
# p.getNumJoints(panda)
# for i in range(p.getNumJoints(panda)):
#     print(p.getJointInfo(panda, i))

# Adjusted pushable object (small cube), positioned on table surface
object_start_position = [0.5, -0.3, 0.65]
object = p.loadURDF("cube_small.urdf", basePosition=object_start_position)
# Adjusted pushable object (small cube), positioned on table surface
# object_start_position2 = [0.5, 0.3, 0.65]
# object2 = p.loadURDF("cube_small.urdf", basePosition=object_start_position2)
# object_start_position3 = [0.7, 0.0, 0.65]
# object3 = p.loadURDF("cube_small.urdf", basePosition=object_start_position3)


# Define and visualize the target area on the table surface
TARGET_POSITION = [0.6, 0.0, 0.625]
target_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.005], rgbaColor=[0, 1, 0, 0.6])
target_body = p.createMultiBody(baseVisualShapeIndex=target_visual, basePosition=TARGET_POSITION)



def add_camera_visualization(position, target_position):
    # Create a small red sphere to represent camera position
    camera_visual = p.createVisualShape(
        p.GEOM_SPHERE, 
        radius=0.03, 
        rgbaColor=[1, 0, 0, 0.8]
    )
    camera_body = p.createMultiBody(
        baseVisualShapeIndex=camera_visual, 
        basePosition=position
    )
    
    # Create a line from camera to target
    p.addUserDebugLine(
        position,
        target_position,
        lineColorRGB=[1, 0, 0],
        lineWidth=2.0
    )
    
    return camera_body

def add_camera_model(position, orientation):
    # You could create a simple camera-shaped object or load a camera model
    # For this example, we'll use a small box with different colors for front/back
    camera_body = p.createVisualShape(
        p.GEOM_BOX, 
        halfExtents=[0.03, 0.05, 0.03], 
        rgbaColor=[0.2, 0.2, 0.2, 0.8]
    )
    lens = p.createVisualShape(
        p.GEOM_CYLINDER, 
        radius=0.02, 
        length=0.03, 
        rgbaColor=[0.1, 0.1, 0.7, 0.8]
    )
    
    camera_id = p.createMultiBody(
        baseVisualShapeIndex=camera_body, 
        basePosition=position,
        baseOrientation=orientation
    )
    
    # Add a lens at the front
    lens_pos = [
        position[0] + 0.04 * orientation[0],
        position[1] + 0.04 * orientation[1], 
        position[2] + 0.04 * orientation[2]
    ]
    p.createMultiBody(
        baseVisualShapeIndex=lens, 
        basePosition=lens_pos,
        baseOrientation=orientation
    )
    
    return camera_id

# Define camera parameters
camera_position = [1.2, 0.0, 1.2]  # Position of the camera
camera_target = [0.7, 0, 0.8]     # Where the camera is looking

# Compute view matrix
view_matrix = p.computeViewMatrix(
    cameraEyePosition=camera_position,
    cameraTargetPosition=camera_target,
    cameraUpVector=[0, 0, 1]
)

# Add visual marker for camera
camera_visual = add_camera_model(camera_position, camera_target)

projection_matrix = p.computeProjectionMatrixFOV(
    fov=60,
    aspect=1.0,
    nearVal=0.01,
    farVal=2.0
)

# Simulation loop
for step in range(100000):
    p.stepSimulation()

    # Capture camera image every 100 steps
    if step % 100 == 0:
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=224, height=224,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )

        # Save camera image to file
        # filename = f"step_{step}.png"
        # cv2.imwrite(filename, rgb_img)

        # [DEBUG only]
        # print("[DEBUG] Step: ", step)
        # print("[DEBUG] RGB image shape: ", rgb_img.shape)
        # print("[DEBUG] Depth image shape: ", depth_img.shape)
        # print("[DEBUG] Segmentation image shape: ", seg_img.shape)

        # print("[DEBUG] Image width x height: ", width, "x", height)


    time.sleep(1./240.)  # Simulate in real-time

# Disconnect after simulation
p.disconnect()
