import pybullet as p
import pybullet_data
import time
import numpy as np

def setup_environment():
    """Set up the basic environment with plane, table, and robot."""
    # Reset simulation
    p.resetSimulation()
    
    # Set gravity
    p.setGravity(0, 0, -9.81)
    
    # Load plane (ground)
    p.loadURDF("plane.urdf", basePosition=[0, 0, 0])
    
    # Load table
    table_position = [0.5, 0, 0]
    table_id = p.loadURDF("table/table.urdf", basePosition=table_position)
    
    # Load robot arm (Panda)
    robot_position = [0, 0, 0.6]
    robot_id = p.loadURDF("franka_panda/panda.urdf", 
                           basePosition=robot_position, 
                           useFixedBase=True)
    
    # Load a test cube to see if the robot can interact with it
    cube_position = [0.5, 0, 0.65]  # Position on the table
    cube_id = p.loadURDF("cube_small.urdf", basePosition=cube_position)
    
    return robot_id, table_id, cube_id

def set_robot_pose(robot_id, joint_positions):
    """Set the robot to a specific pose using joint positions."""
    # Reset joint positions for the 7 controllable joints
    for i in range(7):
        p.resetJointState(robot_id, i, joint_positions[i])
        
        # Set position control
        max_force = 87.0 if i < 4 else 12.0  # First 4 joints have higher force limits
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_positions[i],
            force=max_force
        )
    
    # Set gripper finger joints to close position
    p.resetJointState(robot_id, 9, 0.0)  # Left finger
    p.resetJointState(robot_id, 10, 0.0)  # Right finger
    
    # Control gripper to stay closed
    for i in [9, 10]:
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0.0,  # Fingers closed
            force=20.0
        )

def print_end_effector_info(robot_id):
    """Print information about the end effector position."""
    # End effector is typically link 11 for Panda
    end_effector_index = 11
    state = p.getLinkState(robot_id, end_effector_index)
    print(f"End effector position: {state[0]}")
    print(f"End effector orientation: {state[1]}")
    
    # Print the height above the table (assuming table top is at z=0.625)
    table_height = 0.625
    height_above_table = state[0][2] - table_height
    print(f"Height above table: {height_above_table:.4f} m")

def main():
    # Connect to PyBullet GUI
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    
    # Setup environment
    robot_id, table_id, cube_id = setup_environment()
    
    # Poses to test
    poses = [
        # Format: [joint1, joint2, joint3, joint4, joint5, joint6, joint7]
        # Initial pose from your original code
        [0, -0.3, 0, -2.2, 0, 2.0, 0.8],  # Initial pose
        [-0.156, -0.058, 0.094, -2.447, -0.062, 2.444, 0.843],   # Pre pushing position
        [0.0, 0.347, -0.0, -2.431, -0.0, 2.732, 0.625],  # Good pushing position
        [0.0, 0.347, -0.0, -2.431, -0.0, 2.85, 0.78],  # Good pushing position v2
    ]
    
    current_pose = 0
    
    print("\n===== ROBOT POSE TESTING =====")
    print("Press 'n' to cycle through different poses")
    print("Press 'q' to quit\n")
    
    # Set initial pose
    set_robot_pose(robot_id, poses[current_pose])
    print(f"Testing Pose {current_pose + 1}/{len(poses)}")
    print_end_effector_info(robot_id)
    
    # Main loop
    while True:
        # Step simulation
        p.stepSimulation()
        
        # Get key events
        keys = p.getKeyboardEvents()
        
        # 'n' key pressed - next pose
        if 110 in keys and keys[110] & p.KEY_WAS_TRIGGERED:  # 110 is ASCII for 'n'
            current_pose = (current_pose + 1) % len(poses)
            set_robot_pose(robot_id, poses[current_pose])
            print(f"\nTesting Pose {current_pose + 1}/{len(poses)}")
            print(f"Joint positions: {poses[current_pose]}")
            print_end_effector_info(robot_id)
            
            # Optional: Add debug lines to visualize end effector path for pushing
            ee_state = p.getLinkState(robot_id, 11)
            ee_pos = ee_state[0]
            p.addUserDebugLine(
                ee_pos,
                [ee_pos[0] + 0.2, ee_pos[1], ee_pos[2]],  # 20cm line in x direction
                lineColorRGB=[1, 0, 0],
                lineWidth=2.0,
                lifeTime=5  # Line disappears after 5 seconds
            )
        
        # 'q' key pressed - quit
        if 113 in keys and keys[113] & p.KEY_WAS_TRIGGERED:  # 113 is ASCII for 'q'
            break
        
        time.sleep(0.01)
    
    # Disconnect
    p.disconnect()

if __name__ == "__main__":
    main()