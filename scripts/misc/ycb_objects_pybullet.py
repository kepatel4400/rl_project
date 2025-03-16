import os
import time
import pybullet as p
import pybullet_data
from urdf_models import models_data
import random

# initialize the GUI and others
p.connect(p.GUI)
p.resetDebugVisualizerCamera(3, 90, -30, [0.0, -0.0, -0.0])
p.setTimeStep(1 / 240.)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# load urdf data
models = models_data.model_lib()

# load model list
namelist = models.model_name_list
print("Look at what we have {}".format(namelist))

# Load table and plane
p.loadURDF("plane.urdf")
p.loadURDF("table/table.urdf")

# load the randomly picked model
flags = p.URDF_USE_INERTIA_FROM_FILE


desired_objects = ['cracker_box', 'sugar_box', "soap", "pudding_box", "gelatin_box", "blue_tea_box"]

# randomly get a model
# for i in range(8):
#     random_model = namelist[random.randint(0, len(namelist) - 1)]
#     p.loadURDF(models[random_model], [0., 0., 0.8 + 0.15*i], flags=flags)

# get the desired objects and give them unique positions on XY plane
for i, obj in enumerate(desired_objects):
    p.loadURDF(models[obj], [0.5, 0.3 * (i - 4), 0.8], flags=flags)

p.setGravity(0, 0, -9.8)

while 1:
    p.stepSimulation()
    time.sleep(1./240)