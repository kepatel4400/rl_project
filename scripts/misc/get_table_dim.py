import pybullet as p
import pybullet_data
import time
import numpy as np

# Connect to PyBullet in GUI mode
p.connect(p.GUI)

# Set the search path for PyBullet data assets
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the table URDF into the simulation
table_id = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0])

# Get basic AABB dimensions
aabb = p.getAABB(table_id)
aabb_length = aabb[1][0] - aabb[0][0]  # X-axis dimension
aabb_width = aabb[1][1] - aabb[0][1]   # Y-axis dimension
aabb_height = aabb[1][2] - aabb[0][2]  # Z-axis dimension

print(f"AABB Table Dimensions (L x W x H): {aabb_length:.3f} x {aabb_width:.3f} x {aabb_height:.3f} meters")

# Get more detailed information using visual shape data
visual_data = p.getVisualShapeData(table_id)
print("\nVisual Shape Data:")
for i, shape in enumerate(visual_data):
    shape_id, link_index, type_id, dimensions, filename, local_pos, local_orn, rgba_color = shape
    print(f"Shape {i}: Link {link_index}, Type {type_id}, Dimensions {dimensions}")
    print(f"   Position: {local_pos}")

# Find the tabletop by analyzing visual shape data
# Usually, the tabletop is the shape with:
# 1. Largest horizontal area (length * width)
# 2. Relatively small height compared to length/width

tabletop_candidates = []
for i, shape in enumerate(visual_data):
    shape_id, link_index, type_id, dimensions, filename, local_pos, local_orn, rgba_color = shape
    length, width, height = dimensions
    
    # Check if this shape has dimensions that suggest it's a tabletop
    # (wider and longer than it is tall)
    if length > height*2 and width > height*2:
        area = length * width
        tabletop_candidates.append((i, dimensions, local_pos, area))

# Sort candidates by area (largest first)
if tabletop_candidates:
    tabletop_candidates.sort(key=lambda x: x[3], reverse=True)
    tabletop_idx, tabletop_dims, tabletop_pos, _ = tabletop_candidates[0]
    
    tabletop_length, tabletop_width, tabletop_height = tabletop_dims
    
    print(f"\nTabletop Dimensions (L x W x H): {tabletop_length:.3f} x {tabletop_width:.3f} x {tabletop_height:.3f} meters")
    print(f"Tabletop Position: {tabletop_pos}")
    print(f"This corresponds to Shape {tabletop_idx} in the Visual Shape Data")

    # Create a visual marker for the tabletop dimensions
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[tabletop_length/2, tabletop_width/2, 0.001],  # Thin box to highlight surface
        rgbaColor=[1, 0, 0, 0.3]  # Red, semi-transparent
    )
    
    # Adjust position to place marker right on the tabletop surface
    marker_pos = [0, 0, tabletop_height/2 + 0.001]  # Slightly above table surface
    
    # Create marker
    marker_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=marker_pos
    )
    
    # Print actual surface dimensions for use in your reinforcement learning setup
    print("\nRecommended dimensions for your RL project:")
    print(f"Surface Length (X): {tabletop_length} meters")
    print(f"Surface Width (Y): {tabletop_width} meters")
    print(f"Height from ground (Z): {tabletop_height} meters")
    print(f"Table center position: {tabletop_pos}")
    
else:
    print("\nCouldn't identify the tabletop specifically.")

# Add axes to visualize coordinate system
axis_length = 0.2
p.addUserDebugLine([0, 0, 0], [axis_length, 0, 0], [1, 0, 0], 2)  # X-axis (red)
p.addUserDebugLine([0, 0, 0], [0, axis_length, 0], [0, 1, 0], 2)  # Y-axis (green)
p.addUserDebugLine([0, 0, 0], [0, 0, axis_length], [0, 0, 1], 2)  # Z-axis (blue)

# Run simulation for visualization
for _ in range(1000000):  # Reduced time for quicker execution
    time.sleep(1.0 / 240.0)
    p.stepSimulation()

# Disconnect when done
p.disconnect()