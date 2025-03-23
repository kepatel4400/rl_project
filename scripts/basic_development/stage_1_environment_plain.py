import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load table and object
p.loadURDF("plane.urdf")
table = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0])
cube = p.loadURDF("cube.urdf", basePosition=[0.5, 0.1, 0.7], globalScaling=0.1)

# Add a target circle (visual)
p.addUserDebugText("Target", [0.6, 0.0, 0.75], textColorRGB=[0, 1, 0], textSize=2)

# # Keep GUI open
time.sleep(100)