# DWA -With Visualization for Robot Trajectory ,USR Path,Reference Path and Realignment Feature 22/04
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion, TransformStamped, Vector3
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from tf2_ros import TransformListener, Buffer, LookupException, ConnectivityException, ExtrapolationException
import math
import numpy as np
from collections import deque
from rclpy.time import Time

class DWAController(Node):
    def __init__(self):
        super().__init__('dwa_controller')
        
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
        
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.sub_cmd_vel = self.create_subscription(Twist, '/cmd_vel_usr', self.cmd_vel_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, '/rosbot_xl_base_controller/odom', self.odom_callback, 10)
        
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_user_path = self.create_publisher(Path, '/user_path', 10)
        self.pub_ref_path = self.create_publisher(Path, '/reference_path', 10)  # For reference path
        
        # NEW: Add path history publisher
        self.pub_robot_history = self.create_publisher(Path, '/robot_path_history', 10)
        
        # TF buffer for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.global_frame = "odom"  
        self.transform_available = False
        
        # Define simulation times explicitly
        self.short_sim_time = 0.5  # Short-term simulation time (for collision avoidance)
        self.user_sim_time = 0.5   # User trajectory simulation time
        self.reference_sim_time = 3.0  # Longer time for reference trajectory
        
        # Timer to check available frames and use the most appropriate one
        self.create_timer(1.0, self.check_available_frames)
        
        self.current_obstacles = []
        self.user_cmd = Twist()
        self.user_trajectory = []  # Store user's intended path
        self.dwa_active = False
        self.robot_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}  # Current robot pose

        # NEW: Path history variables
        self.path_history = deque(maxlen=500)  # Store positions with timestamps
        self.path_history_duration = 5.0  # Duration in seconds to keep path history
        self.path_history_update_rate = 0.1  # Update history every 0.1 seconds
        
        # Create timer for path history updates
        self.create_timer(self.path_history_update_rate, self.update_path_history)

        # New variables for persistent reference trajectory
        self.reference_trajectory = []  # Store the persistent reference trajectory
        self.reference_frame_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}  # The pose when reference was created
        self.reference_index = 0  # Current index in reference trajectory (for sliding window)
        self.reference_valid = False
        self.ref_input_time = self.get_clock().now()
        self.ref_update_interval = 0.5  # Update reference every 0.5 second 

        # New flag to track if there's a valid user input
        self.has_user_input = False
        self.last_input_time = self.get_clock().now()
        self.input_timeout = 0.5  # 1 second timeout for input

        # Parameters
        self.robot_radius = 0.35
        self.safety_margin = 0.2
        self.obstacle_threshold = 0.7  # Default obstacle threshold
        self.realignment_obstacle_threshold = 0.5  # Smaller threshold during realignment
        self.angle_threshold = math.radians(60) / 2
        self.dt = 0.1
        self.v_max = 0.7
        self.w_max = 1.2
        self.acc_lin = self.v_max / 0.1
        self.acc_ang = self.w_max / 0.1

        # Scoring weights
        self.trajectory_weight = 3.0
        self.clearance_weight = 2.0
        self.velocity_weight = 1.0
        self.angular_weight = 1.5

        # Current state
        self.current_v = 0.0
        self.current_w = 0.0
        self.mode = "ACCEL"
        
        # Realignment feature variables
        self.realignment_state = "NORMAL"  # States: NORMAL, OBSTACLE_DETECTED, REALIGNING
        self.obstacle_detected_time = self.get_clock().now()
        self.saved_reference_trajectory = []  # Snapshot of reference trajectory at obstacle detection
        self.saved_reference_frame_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}  # Pose at obstacle detection
        self.original_direction = 0.0  # Theta at obstacle detection
        self.realignment_target_index = 0  # Target index in saved reference trajectory
        self.realignment_timeout = 3.5  # Timeout for realignment mode (seconds)
        self.realignment_start_time = self.get_clock().now()
        
        # Thresholds for realignment completion
        self.angle_realignment_threshold = math.radians(5.0)  # 6 degrees tolerance
        self.path_realignment_threshold = 0.02 # 5cm tolerance
        self.window_size_for_target = 10  # Number of points to use as target for realignment
        
        # Create state monitor timer (publish every 0.5 seconds)
        self.create_timer(0.5, self.monitor_realignment_state)

    # NEW: Method to update the robot's path history
    def update_path_history(self):
        """Update the path history with the current robot pose"""
        if not self.transform_available:
            return
            
        # Get current time
        current_time = self.get_clock().now()
        
        # Add current pose to history with timestamp
        self.path_history.append({
            'x': self.robot_pose['x'],
            'y': self.robot_pose['y'],
            'theta': self.robot_pose['theta'],
            'time': current_time
        })
        
        # Prune old positions (older than path_history_duration)
        self.prune_path_history()
        
        # Publish the path history
        self.publish_path_history()
    
    # NEW: Method to prune old positions from the path history
    def prune_path_history(self):
        """Remove positions older than path_history_duration from the path history"""
        if not self.path_history:
            return
            
        current_time = self.get_clock().now()
        
        # Keep removing the oldest position until all positions are within the duration
        while self.path_history:
            oldest_pose = self.path_history[0]
            time_diff = (current_time - oldest_pose['time']).nanoseconds / 1e9
            
            if time_diff > self.path_history_duration:
                self.path_history.popleft()
            else:
                break
    
    # NEW: Method to publish the path history
    def publish_path_history(self):
        """Publish the robot's path history as a Path message"""
        if not self.path_history:
            return
            
        # Create path message
        path_msg = Path()
        path_msg.header.frame_id = self.global_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Add each position to the path
        for pose in self.path_history:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = pose['x']
            pose_stamped.pose.position.y = pose['y']
            
            pose_stamped.pose.orientation.x = 0.0
            pose_stamped.pose.orientation.y = 0.0
            pose_stamped.pose.orientation.z = math.sin(pose['theta'] / 2.0)
            pose_stamped.pose.orientation.w = math.cos(pose['theta'] / 2.0)
            
            path_msg.poses.append(pose_stamped)
        
        # Publish the path
        self.pub_robot_history.publish(path_msg)
        
        # Debug info if needed
        if len(self.path_history) % 50 == 0:  # Log every 50 updates to avoid spam
            self.get_logger().debug(f'Published robot path history with {len(path_msg.poses)} points')

    def check_available_frames(self):
        """Check which frames are available and use the most appropriate one"""
        try:
            # Try to get transform from base_link to odom - this should always work on ROS2 navigation
            transform = self.tf_buffer.lookup_transform(
                "odom",
                'base_link',
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)  # Short timeout
            )
            
            # Check if the transform is valid (not identity)
            trans_x = transform.transform.translation.x
            trans_y = transform.transform.translation.y
            
            # Print the transform to help debug
            self.get_logger().info(f'Transform from base_link to odom: ({trans_x:.3f}, {trans_y:.3f})')
            
            # Check if the transform is zero or nearly zero 
            if abs(trans_x) < 0.001 and abs(trans_y) < 0.001:
                self.get_logger().warning('Transform appears to be identity - using base_link frame')
                self.global_frame = "base_link"
                self.transform_available = False
            else:
                self.global_frame = "odom"
                self.transform_available = True
                
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warning(f'Failed to get transform: {e}')
            self.global_frame = "base_link"
            self.transform_available = False

    def odom_callback(self, msg):
        # Update robot's current pose from odometry
        self.robot_pose['x'] = msg.pose.pose.position.x
        self.robot_pose['y'] = msg.pose.pose.position.y
        
        # Extract orientation (yaw) from quaternion using basic formula
        orientation_q = msg.pose.pose.orientation
        # Convert quaternion to Euler angles - yaw (simplified for 2D)
        siny_cosp = 2.0 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1.0 - 2.0 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.robot_pose['theta'] = yaw

    def scan_callback(self, msg):
        # Check if there's a recent user input before processing obstacles
        current_time = self.get_clock().now()
        time_since_last_input = (current_time - self.last_input_time).nanoseconds / 1e9

        if time_since_last_input > self.input_timeout:
            # Reset user input flag if no input for more than timeout period
            self.has_user_input = False
            
            # Reset realignment state if user input stops - regardless of current state
            if self.realignment_state != "NORMAL":
                self.get_logger().info("User input stopped - returning to normal mode")
                self.realignment_state = "NORMAL"
                # Clear saved trajectories 
                self.saved_reference_trajectory = []

        # Only process obstacles if there's a valid user input
        if not self.has_user_input:
            self.current_obstacles = []
            self.dwa_active = False
            # Additional safety to so  we're in NORMAL state without user input
            self.realignment_state = "NORMAL"
            return

        # Store previous obstacle count for state transitions
        previous_obstacle_count = len(self.current_obstacles)
        
        # Process obstacle data - select appropriate threshold based on state
        # Use smaller threshold during realignment for better obstacle detection
        detection_threshold = self.realignment_obstacle_threshold if self.realignment_state == "REALIGNING" else self.obstacle_threshold
        
        # Clear current obstacles list before updating
        self.current_obstacles = []
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        center_angle = math.pi / 2
        angle_threshold = self.angle_threshold

        # Process scan data to detect obstacles
        for i, r in enumerate(msg.ranges):
            if r == float('inf'):
                continue
            angle = angle_min + i * angle_increment
            if (center_angle - angle_threshold) <= angle <= (center_angle + angle_threshold) and r < detection_threshold:
                x_lidar = r * math.cos(angle)
                y_lidar = r * math.sin(angle)
                x_robot = y_lidar
                y_robot = -x_lidar
                self.current_obstacles.append((x_robot, y_robot))
        
        # Update current obstacle count and DWA active flag
        current_obstacle_count = len(self.current_obstacles)
        self.dwa_active = current_obstacle_count > 0
        
        # Log obstacles during realignment for debugging
        if self.realignment_state == "REALIGNING" and current_obstacle_count > 0:
            self.get_logger().info(f"Detected {current_obstacle_count} obstacles during realignment")
        
        # State transitions for realignment
        # Only handle state transitions if we're not already in REALIGNING state
        if self.realignment_state == "NORMAL" and current_obstacle_count > 0 and previous_obstacle_count == 0:
            # Transition to OBSTACLE_DETECTED state when we first see an obstacle
            self.realignment_state = "OBSTACLE_DETECTED"
            self.obstacle_detected_time = current_time
            
            # Save the current reference trajectory and robot direction
            if self.reference_valid:
                self.saved_reference_trajectory = self.reference_trajectory.copy()
                self.saved_reference_frame_pose = self.reference_frame_pose.copy()
                self.original_direction = self.robot_pose['theta']
                self.get_logger().info(f"Obstacle detected - saved reference trajectory and direction {self.original_direction:.2f}")
            else:
                self.get_logger().warning("Obstacle detected but no valid reference trajectory available")
                self.realignment_state = "NORMAL"  # Revert to normal if no reference trajectory
        
        # Check if obstacle has been avoided to transition from OBSTACLE_DETECTED to REALIGNING
        elif self.realignment_state == "OBSTACLE_DETECTED" and current_obstacle_count == 0 and previous_obstacle_count > 0:
            # Only transition to REALIGNING if we have a saved reference trajectory
            if self.saved_reference_trajectory:
                self.realignment_state = "REALIGNING"
                # Set target to last window of points in saved trajectory
                end_index = len(self.saved_reference_trajectory)
                start_index = max(0, end_index - self.window_size_for_target)
                self.realignment_target_index = start_index
                # Reset realignment start time for timeout monitoring
                self.realignment_start_time = current_time
                self.get_logger().info(f"Obstacle avoided - entering realignment mode targeting indices {start_index}-{end_index}")
                
                # Log the current angle and target angle for debugging
                current_theta = self.robot_pose['theta']
                self.get_logger().info(f"Current angle: {math.degrees(current_theta):.2f}°, Target angle: {math.degrees(self.original_direction):.2f}°")
                self.get_logger().info(f"Angle difference: {math.degrees(self.get_shortest_angle_diff(self.original_direction, current_theta)):.2f}°")
            else:
                self.realignment_state = "NORMAL"  # Revert to normal if no saved reference

    def cmd_vel_callback(self, msg):
        # Update the last input time and set has_user_input flag
        self.last_input_time = self.get_clock().now()
        self.has_user_input = True

        # Check if the input is actually a command to move
        if abs(msg.linear.x) > 1e-3 or abs(msg.angular.z) > 1e-3:
            self.user_cmd = msg
            
            # Handle reference trajectory updates based on realignment state
            current_time = self.get_clock().now()
            time_since_ref_update = (current_time - self.ref_input_time).nanoseconds / 1e9
            
            input_changed = self.reference_valid and (
                abs(msg.linear.x - self.user_cmd.linear.x) > 0.1 or 
                abs(msg.angular.z - self.user_cmd.angular.z) > 0.1
            )
            
            # Only update reference trajectory in NORMAL mode, not during realignment
            if self.realignment_state == "NORMAL" and (not self.reference_valid or 
                                                      time_since_ref_update > self.ref_update_interval or 
                                                      input_changed):
                # Reset reference frame pose to current robot pose
                self.reference_frame_pose = self.robot_pose.copy()
                self.ref_input_time = current_time
                self.reference_index = 0
                
                # Generate a longer reference trajectory
                self.reference_trajectory = self.simulate_reference_trajectory(
                    msg.linear.x,
                    msg.angular.z
                )
                self.reference_valid = True
                self.get_logger().info("Created new reference trajectory")
            
            # Always update the user trajectory for visualization of current input
            self.user_trajectory = self.simulate_user_trajectory(
                msg.linear.x,
                msg.angular.z
            )
            
            # Publish user and reference trajectories
            self.publish_user_trajectory()
            
            # Publish appropriate reference trajectory based on realignment state
            if self.realignment_state == "REALIGNING":
                self.publish_saved_reference_trajectory()
            else:
                self.publish_reference_trajectory()
            
        else:
            # If input is essentially zero, reset the input flag
            self.has_user_input = False
            self.user_trajectory = []
            # But we keep the reference trajectory valid - we'll only reset when we get a new command

    def simulate_user_trajectory(self, v, w):
        """Generate path based on user velocity command for the user simulation time"""
        num_steps = int(self.user_sim_time / self.dt)
        x, y, theta = 0.0, 0.0, 0.0
        trajectory = []
        for _ in range(num_steps):
            theta += w * self.dt
            x += v * math.cos(theta) * self.dt
            y += v * math.sin(theta) * self.dt
            trajectory.append((x, y, theta, v, w))  # Store position, orientation, and velocities
        return trajectory

    def simulate_reference_trajectory(self, v, w):
        """Generate a longer reference trajectory for persistent navigation"""
        num_steps = int(self.reference_sim_time / self.dt)
        x, y, theta = 0.0, 0.0, 0.0
        trajectory = []
        for _ in range(num_steps):
            theta += w * self.dt
            x += v * math.cos(theta) * self.dt
            y += v * math.sin(theta) * self.dt
            trajectory.append((x, y, theta, v, w))  # Store position, orientation, and velocities
        return trajectory

    def transform_local_to_global(self, local_x, local_y, ref_pose):
        """Transform a single point from local coordinates to global"""
        # Rotate and translate based on reference pose
        rot_angle = ref_pose['theta']
        global_x = local_x * math.cos(rot_angle) - local_y * math.sin(rot_angle) + ref_pose['x']
        global_y = local_x * math.sin(rot_angle) + local_y * math.cos(rot_angle) + ref_pose['y']
        return global_x, global_y

    def transform_global_to_local(self, global_x, global_y, ref_pose):
        """Transform a single point from global coordinates to local relative to ref_pose"""
        # Translate to origin of ref_pose and rotate by negative of ref_pose angle
        dx = global_x - ref_pose['x']
        dy = global_y - ref_pose['y']
        rot_angle = -ref_pose['theta']
        local_x = dx * math.cos(rot_angle) - dy * math.sin(rot_angle)
        local_y = dx * math.sin(rot_angle) + dy * math.cos(rot_angle)
        return local_x, local_y

    def transform_local_trajectory_to_global(self, trajectory, ref_pose=None):
        """Transform trajectory points from local to global frame using a reference pose"""
        # If global frame is base_link or transform is not available, return local trajectory
        if self.global_frame == "base_link" or not self.transform_available:
            return trajectory
            
        # Use current robot pose if no reference pose provided
        if ref_pose is None:
            ref_pose = self.robot_pose
            
        try:
            # For debugging
            if trajectory:
                self.get_logger().debug(f'Transforming trajectory using ref pose: ({ref_pose["x"]:.3f}, {ref_pose["y"]:.3f}, {ref_pose["theta"]:.3f})')
            
            # Transform each point
            global_trajectory = []
            for x, y, theta, v, w in trajectory:
                # Transform point using reference pose
                global_x, global_y = self.transform_local_to_global(x, y, ref_pose)
                global_theta = theta + ref_pose['theta']
                
                # Normalize angle
                while global_theta > math.pi:
                    global_theta -= 2 * math.pi
                while global_theta <= -math.pi:
                    global_theta += 2 * math.pi
                    
                global_trajectory.append((global_x, global_y, global_theta, v, w))
            
            return global_trajectory
            
        except Exception as e:
            self.get_logger().warning(f'Transform failed: {e}')
            return trajectory
            
    def publish_user_trajectory(self):
        """Publish the user's intended trajectory as a Path message in global frame"""
        if not self.user_trajectory:
            return
            
        # Transform trajectory to global frame - using current robot pose
        global_trajectory = self.transform_local_trajectory_to_global(self.user_trajectory)
        if not global_trajectory:
            self.get_logger().warning('Failed to get global trajectory')
            return
            
        # Create path message
        path_msg = Path()
        path_msg.header.frame_id = self.global_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for x, y, theta, _, _ in global_trajectory:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = math.sin(theta / 2.0)
            pose.pose.orientation.w = math.cos(theta / 2.0)
            
            path_msg.poses.append(pose)
            
        self.pub_user_path.publish(path_msg)
        
        # Debug info
        self.get_logger().debug(f'Published user trajectory with {len(path_msg.poses)} points in {self.global_frame} frame')

    def publish_reference_trajectory(self):
        """Publish the reference trajectory as a Path message in global frame"""
        if not self.reference_valid or not self.reference_trajectory:
            return
            
        global_trajectory = self.transform_local_trajectory_to_global(
            self.reference_trajectory, 
            self.reference_frame_pose
        )
        
        if not global_trajectory:
            self.get_logger().warning('Failed to get global reference trajectory')
            return
            
        # Create path message
        path_msg = Path()
        path_msg.header.frame_id = self.global_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for x, y, theta, _, _ in global_trajectory:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = math.sin(theta / 2.0)
            pose.pose.orientation.w = math.cos(theta / 2.0)
            
            path_msg.poses.append(pose)
            
        self.pub_ref_path.publish(path_msg)
        
        # Debug info
        self.get_logger().debug(f'Published reference trajectory with {len(path_msg.poses)} points in {self.global_frame} frame')

    # Method to publish the saved reference trajectory during realignment
    def publish_saved_reference_trajectory(self):
        """Publish the saved reference trajectory for realignment"""
        if not self.saved_reference_trajectory:
            return
            
        # Transform trajectory to global frame - using saved reference frame pose
        global_trajectory = self.transform_local_trajectory_to_global(
            self.saved_reference_trajectory, 
            self.saved_reference_frame_pose
        )
        
        if not global_trajectory:
            self.get_logger().warning('Failed to get global saved reference trajectory')
            return
            
        # Create path message
        path_msg = Path()
        path_msg.header.frame_id = self.global_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for x, y, theta, _, _ in global_trajectory:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = math.sin(theta / 2.0)
            pose.pose.orientation.w = math.cos(theta / 2.0)
            
            path_msg.poses.append(pose)
            
        self.pub_ref_path.publish(path_msg)
        
        # Debug info
        self.get_logger().debug(f'Published saved reference trajectory with {len(path_msg.poses)} points in {self.global_frame} frame')

    def get_reference_segment(self):
        """Get the current sliding window portion of the reference trajectory"""
        if not self.reference_valid or not self.reference_trajectory:
            return []
            
        # Check if we have enough trajectory left
        window_size = int(self.short_sim_time / self.dt)
        if self.reference_index + window_size > len(self.reference_trajectory):
            # Not enough trajectory left - could reset or just use what's left
            remaining = len(self.reference_trajectory) - self.reference_index
            self.get_logger().debug(f'Reference trajectory near end - only {remaining} points left')
            
            if remaining < 5:  # If very few points left, reset index to use the beginning again
                self.reference_index = 0
                
        # Extract the window
        end_index = min(self.reference_index + window_size, len(self.reference_trajectory))
        return self.reference_trajectory[self.reference_index:end_index]

    # Method to get a segment of the saved reference trajectory during realignment
    def get_realignment_target_segment(self):
        """Get the current target segment from saved reference trajectory for realignment"""
        if not self.saved_reference_trajectory:
            return []
            
        # Define the window size for the target segment
        window_size = int(self.short_sim_time / self.dt)
        
        # For realignment, to target a point farther along the trajectory last 10 points
        # To robot back to the original path 
        target_index = min(len(self.saved_reference_trajectory) - 1, 
                          self.realignment_target_index + window_size * 2)
        
        # Create a segment that extends from current index to target index
        if self.realignment_target_index >= len(self.saved_reference_trajectory):
            self.realignment_target_index = 0
        
        # Extract the window from saved trajectory
        end_index = min(target_index + 1, len(self.saved_reference_trajectory))
        start_index = max(0, end_index - window_size)
        
        target_segment = self.saved_reference_trajectory[start_index:end_index]
        
        self.get_logger().debug(f"Realignment target segment: {start_index} to {end_index}")
        return target_segment

    def get_dynamic_window(self, current_v, current_w):
        #  dynamic window calculation
        return (
            max(-self.v_max, current_v - self.acc_lin * self.short_sim_time),
            min(self.v_max, current_v + self.acc_lin * self.short_sim_time),
            max(-self.w_max, current_w - self.acc_ang * self.short_sim_time),
            min(self.w_max, current_w + self.acc_ang * self.short_sim_time)
        )

    def simulate_trajectory(self, v_target, w_target, simulation_time=None):
        """Returns (valid, trajectory, min_distance) with constant acceleration"""
        if simulation_time is None:
            simulation_time = self.short_sim_time
            
        num_steps = int(simulation_time / self.dt)
        x, y, theta = 0.0, 0.0, 0.0
        trajectory = []
        min_distance = float('inf')
        valid = True

        # Current velocity
        v = self.current_v
        w = self.current_w

        # Calculate acceleration needed
        if self.mode == "IGNORE_ACCEL":
            #Instant velocity change (ignore acceleration limits)
            v = v_target
            w = w_target
            a_v = 0
            a_w = 0
        else:
            #   acceleration to reach target velocity
            a_v = (v_target - v) / self.short_sim_time
            a_w = (w_target - w) / self.short_sim_time

            # Cap accelerations to limits
            a_v = max(-self.acc_lin, min(self.acc_lin, a_v))
            a_w = max(-self.acc_ang, min(self.acc_ang, a_w))

        for i in range(num_steps):
            if self.mode != "IGNORE_ACCEL":
                # Apply acceleration for each time step
                v += a_v * self.dt
                w += a_w * self.dt

                #  velocities to limits
                v = max(-self.v_max, min(self.v_max, v))
                w = max(-self.w_max, min(self.w_max, w))

            #  position based on acceleration
            theta += w * self.dt
            x += v * math.cos(theta) * self.dt
            y += v * math.sin(theta) * self.dt
            trajectory.append((x, y, theta, v, w))  # Store position, orientation, and velocities

            # Always check for collisions regardless of state
            if v >= 0:  # Only check if moving forward
                for (ox, oy) in self.current_obstacles:
                    dx = ox - x
                    dy = oy - y
                    distance = math.hypot(dx, dy)
                    if distance < (self.robot_radius + self.safety_margin):
                        valid = False
                    min_distance = min(min_distance, distance)
            if not valid:
                break

        return (valid, trajectory, min_distance)

    # Modified to consider realignment state with improved scoring
    def calculate_score(self, candidate_traj, v, min_clearance):
        """Score based on trajectory similarity to reference/realignment trajectory"""
        # Higher weights for realignment - prioritize angle correction
        alignment_weight = 50.0 if self.realignment_state == "REALIGNING" else 3.0
        trajectory_weight = 20.0 if self.realignment_state == "REALIGNING" else self.trajectory_weight
        
        # Increase obstacle clearance weight during realignment for better avoidance
        clearance_weight = self.clearance_weight * 1.5 if self.realignment_state == "REALIGNING" else self.clearance_weight
        
        # If we're in realignment mode, use the saved reference trajectory as target
        if self.realignment_state == "REALIGNING" and self.saved_reference_trajectory:
            target_segment = self.get_realignment_target_segment()
            if target_segment:
                # Calculate components for realignment:
                # 1. Path deviation: how close we're getting to the target path points
                # 2. Direction alignment: how close our direction is to the original direction
                # 3. Optimal rotation: strong bias toward shortest rotation path
                # 4. Obstacle clearance: enhanced weight during realignment
                
                # Only compare the duration that overlaps
                min_len = min(len(target_segment), len(candidate_traj))
                if min_len == 0:
                    return 0.0

                # 1. Path deviation score
                total_deviation = 0.0
                for i in range(min_len):
                    rx, ry, _, _, _ = target_segment[i]
                    cx, cy, _, _, _ = candidate_traj[i]
                    total_deviation += math.hypot(cx - rx, cy - ry)
                
                avg_deviation = total_deviation / min_len
                trajectory_score = -avg_deviation * trajectory_weight
                
                # 2. Direction alignment score
                if len(candidate_traj) > 0:
                    final_theta = candidate_traj[-1][2]  # Get final orientation
                    
                    # Get the shortest angle difference (signed)
                    angle_diff = self.get_shortest_angle_diff(self.original_direction, final_theta)
                    
                    # Calculate direction score based on alignment - penalize larger differences
                    direction_score = -abs(angle_diff) * alignment_weight
                    
                    # 3. Add a bonus for rotation in the correct direction (optimal path)
                    # First determine which way we need to rotate to align
                    needed_rotation = self.get_shortest_angle_diff(self.original_direction, self.robot_pose['theta'])
                    
                    # Check if the trajectory rotates in the needed direction
                    if len(candidate_traj) > 1:
                        start_theta = candidate_traj[0][2]
                        end_theta = candidate_traj[-1][2]
                        trajectory_rotation = self.get_shortest_angle_diff(end_theta, start_theta)
                        
                        # If rotation direction matches needed direction, add bonus
                        if (needed_rotation > 0 and trajectory_rotation > 0) or \
                           (needed_rotation < 0 and trajectory_rotation < 0):
                            # Stronger bonus for trajectories with more rotation in the right direction
                            direction_bonus = abs(trajectory_rotation) * 5.0
                        else:
                            # Significant penalty for rotating the wrong way
                            direction_bonus = -abs(trajectory_rotation) * 8.0
                    else:
                        direction_bonus = 0.0
                else:
                    direction_score = 0.0
                    direction_bonus = 0.0
                
                # 4. Add strong obstacle clearance score - 

                # Lower clearance = higher penalty
                if min_clearance < float('inf'):
                    #  increase penalty as clearance decreases
                    clearance_score = (min_clearance - (self.robot_radius + self.safety_margin)) * clearance_weight
                    #  valid trajectories with minimum clearance still get positive scores
                    clearance_score = max(clearance_score, -20.0)  
                else:
                    # No obstacles in path
                    clearance_score = 0.0
                
                # Combine scores - 
                final_score = trajectory_score + direction_score + direction_bonus + clearance_score
                
                return final_score
        
        if self.reference_valid:
            reference_segment = self.get_reference_segment()
            if reference_segment:
                # Only compare the duration that overlaps
                min_len = min(len(reference_segment), len(candidate_traj))
                if min_len == 0:
                    return 0.0

                total_deviation = 0.0
                for i in range(min_len):
                    rx, ry, _, _, _ = reference_segment[i]
                    cx, cy, _, _, _ = candidate_traj[i]
                    total_deviation += math.hypot(cx - rx, cy - ry)

                avg_deviation = total_deviation / min_len
                trajectory_score = -avg_deviation * self.trajectory_weight
                clearance_score = min_clearance * self.clearance_weight if min_clearance != float('inf') else 0.0
                reverse_bonus = 5.0 if v < 0 else 0.0

                return trajectory_score + clearance_score + reverse_bonus
        
        # If no reference available, revert to original scoring against user trajectory
        if not self.user_trajectory:
            return 0.0

        # Only compare the duration that overlaps between user and candidate trajectories
        min_len = min(len(self.user_trajectory), len(candidate_traj))
        if min_len == 0:
            return 0.0

        total_deviation = 0.0
        for i in range(min_len):
            ux, uy, _, _, _ = self.user_trajectory[i]
            cx, cy, _, _, _ = candidate_traj[i]
            total_deviation += math.hypot(cx - ux, cy - ry)

        avg_deviation = total_deviation / min_len
        trajectory_score = -avg_deviation * self.trajectory_weight
        clearance_score = min_clearance * self.clearance_weight if min_clearance != float('inf') else 0.0
        reverse_bonus = 5.0 if v < 0 else 0.0

        return trajectory_score + clearance_score + reverse_bonus
    
    # Helper function to normalize angle to [-pi, pi]
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle <= -math.pi:
            angle += 2 * math.pi
        return angle

    # Get shortest angle difference (considering direction)
    def get_shortest_angle_diff(self, target_angle, current_angle):
        """Calculate the shortest angle difference, including direction (positive = CCW, negative = CW)"""
        diff = self.normalize_angle(target_angle - current_angle)
        return diff  # Will be positive if CCW is shorter, negative if CW is shorter

    # Check if realignment is complete
    def check_realignment_complete(self):
        """Check if the realignment has been completed"""
        if self.realignment_state != "REALIGNING":
            return False
            
        # Check angle alignment - use absolute difference for threshold check
        angle_diff = self.get_shortest_angle_diff(self.original_direction, self.robot_pose['theta'])
        angle_aligned = abs(angle_diff) < self.angle_realignment_threshold
        
        # If angle is aligned, we can consider realignment complete
        if angle_aligned:
            self.get_logger().info(f"Realignment complete - angle difference: {math.degrees(angle_diff):.2f} degrees")
            self.realignment_state = "NORMAL"
            # Reset saved references
            self.saved_reference_trajectory = []
            return True
            
        # Log realignment progress
        self.get_logger().debug(f"Realignment in progress - angle diff: {math.degrees(angle_diff):.2f} degrees")
        return False

    def compute_best_velocity(self):
        best_score = -float('inf')
        best_v, best_w = self.user_cmd.linear.x, self.user_cmd.angular.z
        best_trajectory = []

        v_min, v_max, w_min, w_max = self.get_dynamic_window(
            self.current_v, self.current_w
        )

        # During realignment, strongly adjust the velocity ranges to favor optimal alignment
        if self.realignment_state == "REALIGNING":
            # Get shortest path direction difference (sign indicates rotation direction)
            angle_diff = self.get_shortest_angle_diff(self.original_direction, self.robot_pose['theta'])
            
            # Log the current angle difference for debugging
            self.get_logger().debug(f"Realignment angle diff: {math.degrees(angle_diff):.2f}° - Current: {math.degrees(self.robot_pose['theta']):.2f}°, Target: {math.degrees(self.original_direction):.2f}°")
            
            # Only adjust velocity ranges if no obstacles nearby - if obstacles present, keep full range
            # for better obstacle avoidance
            if not self.dwa_active:
                # Stronger bias for angular velocity to force optimal turning direction
                if angle_diff > 0:  # Need to turn CCW (positive angular velocity)
                    w_min = max(w_min, 0.0)  # Force only positive angular velocities
                    # Prioritize turning with a minimum angular velocity 
                    if abs(angle_diff) > math.radians(20):
                        w_min = max(w_min, self.w_max * 0.3)  # Set minimum turning rate
                else:  # Need to turn CW (negative angular velocity)
                    w_max = min(w_max, 0.0)  # Force only negative angular velocities
                    # Prioritize turning with a minimum angular velocity 
                    if abs(angle_diff) > math.radians(20):
                        w_max = min(w_max, -self.w_max * 0.3)  # Set minimum turning rate
                
                # Reduce linear velocity during significant turns to improve turning radius
                if abs(angle_diff) > math.radians(30):
                    v_max = min(v_max, self.v_max * 0.7)
            else:
                # If obstacles present during realignment, log this information
                self.get_logger().info(f"Obstacles present during realignment - using full velocity range for avoidance")
            
            self.get_logger().debug(f"Adjusted velocity ranges - v: [{v_min:.2f}, {v_max:.2f}], w: [{w_min:.2f}, {w_max:.2f}]")

        # Search velocity space
        valid_trajectories_found = False
        
        # First pass: evaluate valid trajectories (non-colliding)
        for v in np.linspace(v_min, v_max, 15):
            for w in np.linspace(w_min, w_max, 15):
                valid, trajectory, min_clearance = self.simulate_trajectory(float(v), float(w))
                if valid:
                    valid_trajectories_found = True
                    score = self.calculate_score(trajectory, v, min_clearance)
                    if score > best_score:
                        best_score = score
                        best_v, best_w = float(v), float(w)
                        best_trajectory = trajectory
        
        # If no valid trajectories found, consider reverse trajectories and emergency behaviors
        if not valid_trajectories_found:
            self.get_logger().warning("No valid forward trajectories found - considering reverse")
            
            # Try reverse trajectories to escape
            for v in np.linspace(-self.v_max, min(0.0, v_min), 10):
                for w in np.linspace(w_min, w_max, 15):
                    valid, trajectory, min_clearance = self.simulate_trajectory(float(v), float(w))
                    # For reverse, we're less concerned about validity (we're trying to escape)
                    score = self.calculate_score(trajectory, v, min_clearance)
                    if score > best_score:
                        best_score = score
                        best_v, best_w = float(v), float(w)
                        best_trajectory = trajectory
            
            # If still no good trajectories, emergency stop/rotate in place
            if best_v == self.user_cmd.linear.x and best_w == self.user_cmd.angular.z:
                self.get_logger().warning("Emergency behavior activated - stopping/rotating in place")
                best_v = 0.0
                # Choose rotation direction based on realignment needs if applicable
                if self.realignment_state == "REALIGNING":
                    angle_diff = self.get_shortest_angle_diff(self.original_direction, self.robot_pose['theta'])
                    best_w = self.w_max * 0.5 if angle_diff > 0 else -self.w_max * 0.5
                else:
                    # Default to rotating right to escape
                    best_w = self.w_max * 0.5
        
        # Check if realignment has been completed
        if self.realignment_state == "REALIGNING":
            self.check_realignment_complete()
                        
        # Advance the reference index after computing best velocity
        # This implements the sliding window approach
        if self.reference_valid and self.realignment_state == "NORMAL":
            # Advance by one step per control cycle
            self.reference_index += 1
            # To make robot don't go beyond the trajectory length
            if self.reference_index >= len(self.reference_trajectory):
                self.reference_index = 0
                
        return (best_v, best_w)

    # Monitor realignment state and handle timeouts
    def monitor_realignment_state(self):
        """Monitor realignment state and apply timeouts if needed"""
        # Skip if not in realignment mode
        if self.realignment_state != "REALIGNING":
            return
            
        # Check for timeout
        current_time = self.get_clock().now()
        time_since_realignment_start = (current_time - self.realignment_start_time).nanoseconds / 1e9
        
        # Check if we've been in realignment mode too long (stuck)
        if time_since_realignment_start > self.realignment_timeout:
            self.get_logger().warning(f"Realignment timeout after {self.realignment_timeout} seconds - returning to normal mode")
            self.realignment_state = "NORMAL"
            self.saved_reference_trajectory = []
            
    def publish_cmd_vel(self):
        # Publish command velocity based on DWA or user input
        cmd = Twist()
        
        # Log the current realignment state for debugging
        if self.realignment_state != "NORMAL":
            self.get_logger().info(f"State: {self.realignment_state}, DWA active: {self.dwa_active}, Has input: {self.has_user_input}")
        
        if (self.dwa_active or self.realignment_state == "REALIGNING") and self.has_user_input:
            v, w = self.compute_best_velocity()
            cmd.linear.x = float(v)
            cmd.angular.z = float(w)
            
            # when angle difference is large
            if self.realignment_state == "REALIGNING" and not self.dwa_active:
                angle_diff = self.get_shortest_angle_diff(self.original_direction, self.robot_pose['theta'])
                
                # If angle difference is significant, ensure angular velocity is strong enough
                if abs(angle_diff) > math.radians(20):
                    # Determine rotation direction needed
                    rotation_direction = 1.0 if angle_diff > 0 else -1.0
                    min_w = rotation_direction * self.w_max * 0.9
                    
                    # Check if current w is too small or in wrong direction
                    if (rotation_direction > 0 and cmd.angular.z < min_w) or \
                       (rotation_direction < 0 and cmd.angular.z > min_w):
                        cmd.angular.z = min_w
                        self.get_logger().debug(f"Forcing angular velocity to {cmd.angular.z:.2f} for realignment")
            
            self.current_v = cmd.linear.x
            self.current_w = cmd.angular.z
                
        elif self.has_user_input:
            cmd = self.user_cmd
            self.current_v = cmd.linear.x
            self.current_w = cmd.angular.z
                
        # Publish the appropriate reference trajectory
        if self.realignment_state == "REALIGNING" and self.saved_reference_trajectory:
            self.publish_saved_reference_trajectory()
        elif self.reference_valid:
            self.publish_reference_trajectory()
            
        self.pub_cmd_vel.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = DWAController()
    node.create_timer(0.1, node.publish_cmd_vel)
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
