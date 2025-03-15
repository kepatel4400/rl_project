import pybullet as p
import pybullet_data
import time

# Connect to PyBullet GUI
p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load Plane (ground)
p.loadURDF("plane.urdf")

# Load Table and verify height clearly
table_position = [0.5, 0, 0]
table = p.loadURDF("table/table.urdf", basePosition=table_position)

# Known approximate table surface height (adjust if visually incorrect)
table_surface_z = 0.65

# Load Panda robot arm
robot_base_position = [0, 0, 0.6]  # Correctly above ground
panda = p.loadURDF("franka_panda/panda.urdf", basePosition=robot_base_position, useFixedBase=True)

# Define End-Effector (EEF) link index (known for Panda)
eef_link_index = 11

# Desired initial end-effector position (clearly above the table surface)
initial_eef_position = [0.5, 0.0, table_surface_z + 0.05]  # 5 cm above table

# Compute IK to get initial joint angles for desired EEF pose
initial_joint_angles = p.calculateInverseKinematics(
    panda,
    eef_link_index,
    initial_eef_position,
)

# Set calculated joint angles to robot
for joint_index in range(7):  # first 7 joints for Panda arm
    p.resetJointState(panda, joint_index, initial_joint_angles[joint_index])

# # Visualization aid: small sphere at desired EEF position (for clarity)
# visual_marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, 1])
# p.createMultiBody(baseVisualShapeIndex=visual_shape,
#                   basePosition=initial_eef_position)

# Let the simulation run for visual verification
while True:
    p.stepSimulation()
    # Slow down simulation for clearer visualization
    p.setTimeStep(1./240.)
