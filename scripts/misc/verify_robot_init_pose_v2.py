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

def add_joint_sliders(robot_id):
    """Add sliders for each robot joint."""
    # Joint info and limits for the Panda robot
    joint_sliders = []
    
    # Get number of joints
    num_joints = p.getNumJoints(robot_id)
    print(f"Robot has {num_joints} joints")
    
    # For each joint, create a slider
    for i in range(7):  # First 7 joints are controllable
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        
        # Only add slider for revolute or prismatic joints
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            
            # Some limits might be unreasonable, so let's clip them
            if lower_limit < -10 or lower_limit > 10:
                lower_limit = -3.14
            if upper_limit < -10 or upper_limit > 10:
                upper_limit = 3.14
                
            # Get the current position as default
            current_pos = p.getJointState(robot_id, i)[0]
            
            # Create slider
            slider_id = p.addUserDebugParameter(
                f"Joint {i}: {joint_name}", 
                lower_limit, 
                upper_limit, 
                current_pos
            )
            joint_sliders.append((i, slider_id))
    
    # Add slider for gripper
    gripper_slider = p.addUserDebugParameter("Gripper", 0.0, 0.04, 0.04)
    
    return joint_sliders, gripper_slider

def add_preset_buttons(predefined_positions):
    """Add buttons for preset positions."""
    buttons = {}
    
    # Create a button for each predefined position
    for name in predefined_positions.keys():
        button = p.addUserDebugParameter(f"Go to {name}", 1, 0, 0)
        buttons[name] = button
    
    # Save current position
    save_button = p.addUserDebugParameter("Save Current Position", 1, 0, 0)
    buttons["save"] = save_button
    
    return buttons

def set_robot_position(robot_id, joint_positions):
    """Set robot to specified joint positions."""
    # First reset joint states
    for i, pos in enumerate(joint_positions):
        p.resetJointState(robot_id, i, pos)
    
    # Then also update the motor control to maintain this position
    for i, pos in enumerate(joint_positions):
        # Get appropriate force limits for each joint
        max_force = 87.0 if i < 4 else 12.0  # First 4 joints have higher force limits
        
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=pos,
            force=max_force
        )

def update_sliders_to_match_position(joint_sliders, joint_positions):
    """
    Update the slider positions to match the current robot joint positions.
    This ensures UI stays in sync with robot state.
    
    NOTE: THIS DOES NOT ACTUALLY UPDATE THE VISIBLE SLIDER POSITIONS IN THE UI
    BECAUSE PYBULLET DOESN'T SUPPORT THAT. It only updates our tracking of them.
    """
    # PyBullet doesn't allow updating the visual position of sliders
    # We can only create new ones (which would be disruptive) or update our tracking
    # of the current values to match what we just set the robot to
    
    # Build a dictionary of new values to use when reading sliders
    slider_values = {}
    for i, (joint_idx, slider_id) in enumerate(joint_sliders):
        if i < len(joint_positions):
            slider_values[slider_id] = joint_positions[i]
            
    return slider_values

def print_current_joint_positions(robot_id):
    """Print current joint positions in a format easy to copy."""
    joint_positions = []
    for i in range(7):
        pos = p.getJointState(robot_id, i)[0]
        joint_positions.append(round(pos, 3))
    
    print("\nCurrent Joint Positions:")
    print(f"[{', '.join(str(p) for p in joint_positions)}]")
    
    # Also print end-effector position
    ee_state = p.getLinkState(robot_id, 11)
    ee_pos = ee_state[0]
    ee_orn = p.getEulerFromQuaternion(ee_state[1])
    ee_orn_deg = [a * 180 / np.pi for a in ee_orn]
    
    print(f"End-effector position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    print(f"End-effector orientation (degrees): [{ee_orn_deg[0]:.1f}, {ee_orn_deg[1]:.1f}, {ee_orn_deg[2]:.1f}]")
    
    # Calculate height above table (assuming table surface is at z=0.625)
    table_height = 0.625
    height_above_table = ee_pos[2] - table_height
    print(f"Height above table: {height_above_table*100:.1f} cm")

def main():
    # Connect to PyBullet GUI
    physicsClient = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Define predefined positions
    predefined_positions = {
        "Pre-push pose": [-0.156, -0.058, 0.094, -2.447, -0.062, 2.444, 0.843],
        "Home pose": [0, -0.3, 0, -2.2, 0, 2.0, 0.8],
        "Push pose": [0.0, 0.347, -0.0, -2.431, -0.0, 3.143, 0.78],
        # You can add more positions here as needed
    }
    
    # Setup environment
    robot_id, table_id, cube_id = setup_environment()
    
    # Add joint sliders
    joint_sliders, gripper_slider = add_joint_sliders(robot_id)
    
    # Add preset buttons
    preset_buttons = add_preset_buttons(predefined_positions)
    
    # Track button states
    last_button_values = {name: p.readUserDebugParameter(button_id) 
                          for name, button_id in preset_buttons.items()}
    
    # List to save positions
    saved_positions = []
    
    # Create dictionary to store slider values
    # When using a preset, this will override what we read from the physical sliders
    slider_overrides = {}
    # Flag to indicate we should use overrides (after pressing a preset button)
    use_slider_overrides = False
    # Counter to control how long we use overrides
    override_counter = 0
    
    print("\n===== ROBOT JOINT CONTROL WITH PRE-RECORDED POSITIONS =====")
    print("Use the sliders to control individual joints")
    print("Use the preset buttons to set predefined positions:")
    for name in predefined_positions.keys():
        print(f"  - {name}: {predefined_positions[name]}")
    print("Press 'Save Current Position' to record good configurations")
    print("Press 'q' to quit and display all saved positions\n")
    
    # Main loop
    running = True
    while running:
        # Check preset buttons
        for name, button_id in preset_buttons.items():
            current_value = p.readUserDebugParameter(button_id)
            
            if current_value != last_button_values[name]:
                last_button_values[name] = current_value
                
                if name == "save":
                    # Save current position
                    joint_positions = []
                    for i in range(7):
                        pos = p.getJointState(robot_id, i)[0]
                        joint_positions.append(pos)
                    
                    # Get end-effector position
                    ee_state = p.getLinkState(robot_id, 11)
                    ee_pos = ee_state[0]
                    
                    # Calculate height above table
                    table_height = 0.625
                    height_above_table = ee_pos[2] - table_height
                    
                    # Save position
                    saved_positions.append({
                        'joint_positions': joint_positions,
                        'ee_position': ee_pos,
                        'height_above_table_cm': height_above_table * 100
                    })
                    
                    print(f"\n✅ Saved Position #{len(saved_positions)}")
                    print_current_joint_positions(robot_id)
                else:
                    # Go to predefined position
                    set_robot_position(robot_id, predefined_positions[name])
                    print(f"\n➡️ Set robot to '{name}' position")
                    print_current_joint_positions(robot_id)
                    
                    # Set up override values to ensure sliders don't immediately override our settings
                    slider_overrides = {}
                    for i, (joint_idx, slider_id) in enumerate(joint_sliders):
                        if i < len(predefined_positions[name]):
                            slider_overrides[slider_id] = predefined_positions[name][i]
                    
                    # Enable overrides for a number of iterations to ensure stability
                    use_slider_overrides = True
                    override_counter = 1000  # Use overrides for 100 iterations
        
        # Read joint sliders and apply, but respect any overrides from preset buttons
        for joint_idx, slider_id in joint_sliders:
            # Get position value - either from slider or from our override
            if use_slider_overrides and slider_id in slider_overrides:
                target_pos = slider_overrides[slider_id]
            else:
                # If user has moved a slider significantly, clear its override
                current_slider_val = p.readUserDebugParameter(slider_id)
                if slider_id in slider_overrides:
                    # If user has adjusted the slider by more than a small threshold, use the new value
                    if abs(current_slider_val - slider_overrides[slider_id]) > 0.01:
                        slider_overrides.pop(slider_id)  # Remove override for this slider
                        target_pos = current_slider_val
                    else:
                        target_pos = slider_overrides[slider_id]
                else:
                    target_pos = current_slider_val
            
            # Apply the position
            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=100  # High force for quick movement
            )
        
        # Decrement override counter and clear if necessary
        if use_slider_overrides:
            override_counter -= 1
            if override_counter <= 0:
                use_slider_overrides = False
        
        # Read and apply gripper position
        gripper_pos = p.readUserDebugParameter(gripper_slider)
        for i in [9, 10]:  # Finger joints
            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=gripper_pos,
                force=20.0
            )
        
        # Check for quit key
        keys = p.getKeyboardEvents()
        if 113 in keys and keys[113] & p.KEY_WAS_TRIGGERED:  # 'q' key
            running = False
        
        # Step simulation
        p.stepSimulation()
        time.sleep(0.02)
    
    # Print all saved positions when exiting
    print("\n===== SAVED ROBOT POSITIONS =====")
    if not saved_positions:
        print("No positions were saved.")
    else:
        for i, pos in enumerate(saved_positions):
            print(f"\nPosition #{i+1}:")
            print(f"Joint positions: {[round(j, 3) for j in pos['joint_positions']]}")
            print(f"End-effector position: {[round(p, 3) for p in pos['ee_position']]}")
            print(f"Height above table: {pos['height_above_table_cm']:.1f} cm")
    
    print("\nCopy the joint positions you prefer into your main code.")
    
    # Disconnect
    p.disconnect()

if __name__ == "__main__":
    main()