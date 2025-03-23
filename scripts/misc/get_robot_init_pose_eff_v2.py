import os
import pybullet as p
import pybullet_data
import time
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
from urdf_models import models_data  # Import your existing models

class RobotCalibrationTool:
    """
    A tool for calibrating robot arm positions by manually moving the end effector
    in Cartesian space and recording joint values.
    """
    
    def __init__(self):
        # Connect to PyBullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Disable default GUI to use our custom one
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)  # Use stepping mode
        
        # Set up camera view
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0.5, 0, 0.5]
        )
        
        # Load environment elements
        self.plane_id = p.loadURDF("plane.urdf", basePosition=[0, 0, 0])
        self.table_id = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, 0])
        
        # Load robot
        self.robot_position = [0, 0, 0.6]
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", 
                                  basePosition=self.robot_position, 
                                  useFixedBase=True)
        
        # Set robot parameters
        self.end_effector_index = 11  # panda_grasptarget_hand
        self.num_joints = p.getNumJoints(self.robot_id)
        
        # Initial joint values - start with these from your code
        # self.initial_joint_positions = [0.0, 0.347, -0.0, -2.431, -0.0, 2.85, 0.78]
        self.initial_joint_positions = [0.0, 0.087, 0.0, -3.068, 0.0, 3.227, 0.78]
        
        # Joint limits for the Panda arm (from your code)
        self.joint_limits = [
            (-2.9671, 2.9671),  # panda_joint1
            (-1.8326, 1.8326),  # panda_joint2
            (-2.9671, 2.9671),  # panda_joint3
            (-3.1416, 0.0),     # panda_joint4
            (-2.9671, 2.9671),  # panda_joint5
            (-0.0873, 3.8223),  # panda_joint6
            (-2.9671, 2.9671),  # panda_joint7
        ]
        
        # Reset robot to initial position
        self._reset_robot()
        
        # Set gripper finger joints to closed position
        p.resetJointState(self.robot_id, 9, 0.0)  # Left finger
        p.resetJointState(self.robot_id, 10, 0.0)  # Right finger
        
        # Close the gripper
        for i in [9, 10]:  # Finger joints
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0,  # Fingers closed
                force=20.0
            )
        
        # Load models lib
        self.models = models_data.model_lib()
        
        # Add test object - using the pudding_box as in your code
        self.object_position = [0.55, 0.0, 0.65]
        yaw = np.pi/2
        self.object_orientation = p.getQuaternionFromEuler([0, 0, yaw])
        flags = p.URDF_USE_INERTIA_FROM_FILE
        self.object_id = p.loadURDF(self.models["pudding_box"], 
                                  basePosition=self.object_position,
                                  baseOrientation=self.object_orientation, 
                                  flags=flags)
        
        # Target position visualization
        self.target_position = [0.7, 0.0, 0.625]
        target_visual = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=[0.05, 0.05, 0.005], 
            rgbaColor=[0, 1, 0, 0.6]
        )
        self.target_id = p.createMultiBody(
            baseVisualShapeIndex=target_visual, 
            basePosition=self.target_position
        )
        
        # Get current end effector position for slider initialization
        self.current_ee_pos = p.getLinkState(self.robot_id, self.end_effector_index)[0]
        self.current_ee_orn = p.getLinkState(self.robot_id, self.end_effector_index)[1]
        
        # Create sliders with a wider range for more flexibility
        self.x_range = (0.1, 0.9)    # Range for x-axis
        self.y_range = (-0.4, 0.4)   # Range for y-axis
        self.z_range = (0.4, 1.0)    # Range for z-axis
        
        # Initialize GUI for sliders
        self._setup_gui()
        
        # Add visual indicator for end effector
        self._add_ee_indicator()
        
        # Run the simulation in a separate thread
        self.running = True
        self.simulation_thread = threading.Thread(target=self._run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
    
    def _reset_robot(self):
        """Reset robot to home position using the initial joint positions."""
        for i in range(7):
            p.resetJointState(self.robot_id, i, self.initial_joint_positions[i])
            
            # Set position control with appropriate force limits
            max_force = 87.0 if i < 4 else 12.0  # First 4 joints have higher force limits
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=self.initial_joint_positions[i],
                force=max_force
            )
    
    def _add_ee_indicator(self):
        """Add a visual indicator at the end effector position."""
        self.ee_visual = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=0.02, 
            rgbaColor=[0, 0, 1, 0.7]
        )
        self.ee_indicator = p.createMultiBody(
            baseVisualShapeIndex=self.ee_visual, 
            basePosition=self.current_ee_pos
        )
    
    def _update_ee_indicator(self):
        """Update the position of the end effector indicator."""
        p.resetBasePositionAndOrientation(
            self.ee_indicator,
            self.current_ee_pos,
            [0, 0, 0, 1]
        )
    
    def _setup_gui(self):
        """Create a Tkinter GUI with sliders for controlling the robot."""
        self.root = tk.Tk()
        self.root.title("Robot Calibration Tool")
        
        # Create a frame for controls
        control_frame = ttk.LabelFrame(self.root, text="End Effector Position")
        control_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        # X-axis controls
        ttk.Label(control_frame, text="X Position").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        # X precision control frame
        x_control_frame = ttk.Frame(control_frame)
        x_control_frame.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # X decrement button
        x_dec_btn = ttk.Button(x_control_frame, text="-0.01", width=4, 
                             command=lambda: self._increment_position('x', -0.01))
        x_dec_btn.pack(side="left")
        
        # X slider
        self.x_slider = ttk.Scale(
            x_control_frame, 
            from_=self.x_range[0], 
            to=self.x_range[1], 
            orient="horizontal",
            length=250,
            value=self.current_ee_pos[0]
        )
        self.x_slider.pack(side="left", padx=5)
        
        # X increment button
        x_inc_btn = ttk.Button(x_control_frame, text="+0.01", width=4, 
                             command=lambda: self._increment_position('x', 0.01))
        x_inc_btn.pack(side="left")
        
        # X value entry and display
        x_value_frame = ttk.Frame(control_frame)
        x_value_frame.grid(row=0, column=2, padx=5, pady=5)
        
        # X value entry field
        self.x_var = tk.StringVar(value=f"{self.current_ee_pos[0]:.3f}")
        self.x_entry = ttk.Entry(x_value_frame, textvariable=self.x_var, width=7)
        self.x_entry.pack(side="left", padx=2)
        self.x_entry.bind("<Return>", lambda e: self._on_entry_change('x'))
        
        # Apply button for X
        x_apply_btn = ttk.Button(x_value_frame, text="Apply", width=5,
                               command=lambda: self._on_entry_change('x'))
        x_apply_btn.pack(side="left", padx=2)
        
        # Y-axis controls
        ttk.Label(control_frame, text="Y Position").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        
        # Y precision control frame
        y_control_frame = ttk.Frame(control_frame)
        y_control_frame.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        # Y decrement button
        y_dec_btn = ttk.Button(y_control_frame, text="-0.01", width=4, 
                             command=lambda: self._increment_position('y', -0.01))
        y_dec_btn.pack(side="left")
        
        # Y slider
        self.y_slider = ttk.Scale(
            y_control_frame, 
            from_=self.y_range[0], 
            to=self.y_range[1], 
            orient="horizontal",
            length=250,
            value=self.current_ee_pos[1]
        )
        self.y_slider.pack(side="left", padx=5)
        
        # Y increment button
        y_inc_btn = ttk.Button(y_control_frame, text="+0.01", width=4, 
                             command=lambda: self._increment_position('y', 0.01))
        y_inc_btn.pack(side="left")
        
        # Y value entry and display
        y_value_frame = ttk.Frame(control_frame)
        y_value_frame.grid(row=1, column=2, padx=5, pady=5)
        
        # Y value entry field
        self.y_var = tk.StringVar(value=f"{self.current_ee_pos[1]:.3f}")
        self.y_entry = ttk.Entry(y_value_frame, textvariable=self.y_var, width=7)
        self.y_entry.pack(side="left", padx=2)
        self.y_entry.bind("<Return>", lambda e: self._on_entry_change('y'))
        
        # Apply button for Y
        y_apply_btn = ttk.Button(y_value_frame, text="Apply", width=5,
                               command=lambda: self._on_entry_change('y'))
        y_apply_btn.pack(side="left", padx=2)
        
        # Z-axis controls
        ttk.Label(control_frame, text="Z Position").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        
        # Z precision control frame
        z_control_frame = ttk.Frame(control_frame)
        z_control_frame.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        # Z decrement button
        z_dec_btn = ttk.Button(z_control_frame, text="-0.01", width=4, 
                             command=lambda: self._increment_position('z', -0.01))
        z_dec_btn.pack(side="left")
        
        # Z slider
        self.z_slider = ttk.Scale(
            z_control_frame, 
            from_=self.z_range[0], 
            to=self.z_range[1], 
            orient="horizontal",
            length=250,
            value=self.current_ee_pos[2]
        )
        self.z_slider.pack(side="left", padx=5)
        
        # Z increment button
        z_inc_btn = ttk.Button(z_control_frame, text="+0.01", width=4, 
                             command=lambda: self._increment_position('z', 0.01))
        z_inc_btn.pack(side="left")
        
        # Z value entry and display
        z_value_frame = ttk.Frame(control_frame)
        z_value_frame.grid(row=2, column=2, padx=5, pady=5)
        
        # Z value entry field
        self.z_var = tk.StringVar(value=f"{self.current_ee_pos[2]:.3f}")
        self.z_entry = ttk.Entry(z_value_frame, textvariable=self.z_var, width=7)
        self.z_entry.pack(side="left", padx=2)
        self.z_entry.bind("<Return>", lambda e: self._on_entry_change('z'))
        
        # Apply button for Z
        z_apply_btn = ttk.Button(z_value_frame, text="Apply", width=5,
                               command=lambda: self._on_entry_change('z'))
        z_apply_btn.pack(side="left", padx=2)
        
        # Buttons frame
        button_frame = ttk.Frame(self.root)
        button_frame.pack(padx=10, pady=10, fill="x")
        
        # Get Joint Values button
        self.get_joints_button = ttk.Button(
            button_frame,
            text="Get Joint Values",
            command=self._print_joint_values
        )
        self.get_joints_button.pack(side="left", padx=5)
        
        # Reset Positions button
        self.reset_button = ttk.Button(
            button_frame,
            text="Reset Position",
            command=self._reset_position
        )
        self.reset_button.pack(side="left", padx=5)
        
        # Output frame for joint values
        output_frame = ttk.LabelFrame(self.root, text="Joint Values Output")
        output_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Scrolled text widget for joint output
        self.output_text = tk.Text(output_frame, height=10, width=50)
        self.output_text.pack(padx=5, pady=5, fill="both", expand=True)
        
        # Status frame
        status_frame = ttk.Frame(self.root)
        status_frame.pack(padx=10, pady=5, fill="x")
        
        # Status label for IK solution
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side="left", padx=5)
        
        # Collision indicator
        self.collision_label = ttk.Label(status_frame, text="No Collision", foreground="green")
        self.collision_label.pack(side="right", padx=5)
        
        # Bind events
        self.x_slider.bind("<Motion>", self._on_slider_change)
        self.y_slider.bind("<Motion>", self._on_slider_change)
        self.z_slider.bind("<Motion>", self._on_slider_change)
    
    def _on_slider_change(self, event=None):
        """Handle slider value changes."""
        try:
            # Update labels with current values
            x_val = self.x_slider.get()
            y_val = self.y_slider.get()
            z_val = self.z_slider.get()
            
            # Update entry fields to reflect slider values
            self.x_var.set(f"{x_val:.3f}")
            self.y_var.set(f"{y_val:.3f}")
            self.z_var.set(f"{z_val:.3f}")
            
            # Apply IK to move end effector
            target_position = [x_val, y_val, z_val]
            self._move_end_effector(target_position)
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", foreground="red")
    
    def _increment_position(self, axis, amount):
        """Increment position by a fixed amount (e.g., +/- 0.1 or +/- 0.01)."""
        try:
            if axis == 'x':
                current = self.x_slider.get()
                # For 0.1 increments, round to 0.1; for 0.01 increments, round to 0.01
                precision = 100
                new_val = round((current + amount) * precision) / precision
                new_val = max(self.x_range[0], min(self.x_range[1], new_val))  # Clamp to range
                self.x_slider.set(new_val)
                self.x_var.set(f"{new_val:.3f}")
            elif axis == 'y':
                current = self.y_slider.get()
                precision = 100
                new_val = round((current + amount) * precision) / precision
                new_val = max(self.y_range[0], min(self.y_range[1], new_val))
                self.y_slider.set(new_val)
                self.y_var.set(f"{new_val:.3f}")
            elif axis == 'z':
                current = self.z_slider.get()
                precision = 100
                new_val = round((current + amount) * precision) / precision
                new_val = max(self.z_range[0], min(self.z_range[1], new_val))
                self.z_slider.set(new_val)
                self.z_var.set(f"{new_val:.3f}")
            
            # Get current slider values
            x_val = self.x_slider.get()
            y_val = self.y_slider.get()
            z_val = self.z_slider.get()
            
            # Apply IK to move end effector
            target_position = [x_val, y_val, z_val]
            self._move_end_effector(target_position)
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", foreground="red")
    
    def _on_entry_change(self, axis):
        """Handle manual entry of position values."""
        try:
            if axis == 'x':
                try:
                    new_val = float(self.x_var.get())
                    new_val = max(self.x_range[0], min(self.x_range[1], new_val))  # Clamp to range
                    self.x_slider.set(new_val)
                except ValueError:
                    # Reset to current slider value if input is invalid
                    self.x_var.set(f"{self.x_slider.get():.3f}")
                    return
            elif axis == 'y':
                try:
                    new_val = float(self.y_var.get())
                    new_val = max(self.y_range[0], min(self.y_range[1], new_val))
                    self.y_slider.set(new_val)
                except ValueError:
                    self.y_var.set(f"{self.y_slider.get():.3f}")
                    return
            elif axis == 'z':
                try:
                    new_val = float(self.z_var.get())
                    new_val = max(self.z_range[0], min(self.z_range[1], new_val))
                    self.z_slider.set(new_val)
                except ValueError:
                    self.z_var.set(f"{self.z_slider.get():.3f}")
                    return
            
            # Get current slider values
            x_val = self.x_slider.get()
            y_val = self.y_slider.get()
            z_val = self.z_slider.get()
            
            # Apply IK to move end effector
            target_position = [x_val, y_val, z_val]
            self._move_end_effector(target_position)
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", foreground="red")
    
    def _move_end_effector(self, target_position):
        """Move the end effector to the target position using IK."""
        # Get current end effector orientation
        _, current_orientation = p.getLinkState(self.robot_id, self.end_effector_index)[:2]
        
        # Calculate IK
        joint_positions = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            target_position,
            current_orientation,
            maxNumIterations=100,
            residualThreshold=1e-4
        )
        
        # Check if solution is within joint limits
        valid_solution = True
        clipped_positions = []
        
        for i in range(7):
            pos = joint_positions[i]
            lower, upper = self.joint_limits[i]
            
            if pos < lower or pos > upper:
                valid_solution = False
                pos = max(lower, min(pos, upper))
            
            clipped_positions.append(pos)
        
        if not valid_solution:
            self.status_label.config(text="Warning: IK solution outside joint limits", foreground="orange")
        else:
            self.status_label.config(text="Valid IK solution", foreground="green")
        
        # Apply joint positions
        for i in range(7):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=clipped_positions[i],
                maxVelocity=1.0
            )
        
        # Step simulation to apply changes
        for _ in range(5):
            p.stepSimulation()
        
        # Update current end effector position
        self.current_ee_pos = p.getLinkState(self.robot_id, self.end_effector_index)[0]
        self.current_ee_orn = p.getLinkState(self.robot_id, self.end_effector_index)[1]
        
        # Update end effector indicator
        self._update_ee_indicator()
        
        # Check for collisions
        self._check_collisions()
        
    def _set_position(self, position):
        """Set position to a preset coordinate."""
        # Update sliders
        self.x_slider.set(position[0])
        self.y_slider.set(position[1])
        self.z_slider.set(position[2])
        
        # Update entry fields
        self.x_var.set(f"{position[0]:.3f}")
        self.y_var.set(f"{position[1]:.3f}")
        self.z_var.set(f"{position[2]:.3f}")
        
        # Apply IK to move end effector
        self._move_end_effector(position)
    
    def _check_collisions(self):
        """Check for collisions between robot and other objects."""
        # Check collision with table
        table_contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.table_id)
        
        # Check collision with object
        object_contacts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.object_id)
        
        # Check for self-collisions
        self_collision = False
        for i in range(self.num_joints):
            for j in range(i+2, self.num_joints):  # Skip adjacent links
                if p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id, 
                                     linkIndexA=i, linkIndexB=j):
                    self_collision = True
                    break
            if self_collision:
                break
        
        # Update collision indicator
        if table_contacts or object_contacts or self_collision:
            self.collision_label.config(text="Collision Detected!", foreground="red")
        else:
            self.collision_label.config(text="No Collision", foreground="green")
    
    def _print_joint_values(self):
        """Print current joint values to the output text and terminal."""
        # Get current joint positions
        joint_values = []
        for i in range(7):
            joint_values.append(p.getJointState(self.robot_id, i)[0])
        
        # Format for display
        formatted_joints = [f"{val:.3f}" for val in joint_values]
        joint_str = f"initial_joint_positions = [{', '.join(formatted_joints)}]"
        
        # Display in output text
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"# Current End Effector Position:\n")
        self.output_text.insert(tk.END, f"# X: {self.current_ee_pos[0]:.3f}, Y: {self.current_ee_pos[1]:.3f}, Z: {self.current_ee_pos[2]:.3f}\n\n")
        self.output_text.insert(tk.END, f"# Joint Values for RoboticPushingEnv:\n")
        self.output_text.insert(tk.END, joint_str)
        
        # Also print to terminal for easy copying
        print("\n" + "="*50)
        print("CURRENT JOINT VALUES:")
        print(joint_str)
        print("="*50 + "\n")
    
    def _reset_position(self):
        """Reset the robot to the initial position."""
        self._reset_robot()
        
        # Update current end effector position
        self.current_ee_pos = p.getLinkState(self.robot_id, self.end_effector_index)[0]
        
        # Update sliders without triggering events
        self.x_slider.set(self.current_ee_pos[0])
        self.y_slider.set(self.current_ee_pos[1])
        self.z_slider.set(self.current_ee_pos[2])
        
        # Update entry fields
        self.x_var.set(f"{self.current_ee_pos[0]:.3f}")
        self.y_var.set(f"{self.current_ee_pos[1]:.3f}")
        self.z_var.set(f"{self.current_ee_pos[2]:.3f}")
        
        # Update end effector indicator
        self._update_ee_indicator()
        
        # Clear status
        self.status_label.config(text="Reset to initial position", foreground="black")
        self._check_collisions()
    
    def _run_simulation(self):
        """Run the simulation in a separate thread."""
        while self.running:
            p.stepSimulation()
            time.sleep(1/240.0)  # Simulate at 240 Hz
    
    def run(self):
        """Run the main Tkinter event loop."""
        self.root.mainloop()
        self.running = False  # Stop simulation thread when GUI is closed

if __name__ == "__main__":
    calibration_tool = RobotCalibrationTool()
    calibration_tool.run()