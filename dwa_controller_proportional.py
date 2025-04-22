# Angle Alignment(Based on Proportional Controller) without marker and extended trajecotry
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point, Quaternion, TransformStamped, Vector3
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from tf2_ros import TransformListener, Buffer, LookupException, ConnectivityException, ExtrapolationException
import math
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Float32
from std_msgs.msg import Header

class DWAController(Node):
    def __init__(self):
        super().__init__('dwa_controller')
        
        # Important: set logging level to debug to see more information
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
        
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.sub_cmd_vel = self.create_subscription(Twist, '/cmd_vel_usr', self.cmd_vel_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, '/rosbot_xl_base_controller/odom', self.odom_callback, 10)
        
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_predicted_path = self.create_publisher(Path, '/predicted_path', 10)
        self.pub_user_path = self.create_publisher(Path, '/user_path', 10)
        # Publisher for heading deviation data
        self.pub_heading_deviation = self.create_publisher(Float32, '/heading_deviation', 10)
        
        # TF buffer for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.global_frame = "odom"  
        self.transform_available = False
        
        # Define simulation times explicitly
        self.short_sim_time = 0.5  # Short-term simulation time (for collision avoidance)
        self.user_sim_time = 0.5   # User trajectory simulation time
        self.extended_sim_time = 3.0  # Extended simulation time for visualization
        
        # Timer to check available frames and use the most appropriate one
        self.create_timer(1.0, self.check_available_frames)
        
        self.current_obstacles = []
        self.user_cmd = Twist()
        self.user_trajectory = []  # Store user's intended path
        self.dwa_active = False
        self.robot_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}  # Current robot pose

        # New flag to track if there's a valid user input
        self.has_user_input = False
        self.last_input_time = self.get_clock().now()
        self.input_timeout = 1.0  # 1 second timeout for input

        # Parameters
        self.robot_radius = 0.35
        self.safety_margin = 0.2
        self.obstacle_threshold = 0.7
        self.angle_threshold = math.radians(75) / 2
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
        
        # Heading tracking for obstacle avoidance
        self.initial_heading = 0.0  # Initial heading when obstacle is detected
        self.max_heading_deviation = 0.0  # Maximum heading deviation during avoidance
        self.current_heading_deviation = 0.0  # Current heading deviation
        self.obstacle_detected_time = None  # Time when obstacle was first detected
        self.obstacle_cleared_time = None  # Time when obstacle was cleared
        self.is_recovering = False  # Flag to indicate if we're in recovery mode
        self.recovery_target_heading = 0.0  # Target heading for recovery
        self.original_user_cmd = Twist()  # Original user command before obstacle detection
        
        # State machine for the controller
        self.state = "NORMAL"  # States: NORMAL, AVOIDING, RECOVERING

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
        
        # If we're in AVOIDING state, track heading deviation
        if self.state == "AVOIDING":
            # Calculate heading deviation from initial heading
            self.current_heading_deviation = normalize_angle(yaw - self.initial_heading)
            
            # Update maximum deviation
            if abs(self.current_heading_deviation) > abs(self.max_heading_deviation):
                self.max_heading_deviation = self.current_heading_deviation
                
            # Publish current heading deviation
            deviation_msg = Float32()
            deviation_msg.data = self.current_heading_deviation
            self.pub_heading_deviation.publish(deviation_msg)
            
            # Log significant heading changes for debugging
            if abs(self.current_heading_deviation) > math.radians(10):  # Log every 10 degrees of change
                self.get_logger().info(f'Heading deviation: {math.degrees(self.current_heading_deviation):.1f} degrees')
        
        # If we're in RECOVERING state, check if we've reached the target heading
        elif self.state == "RECOVERING":
            # Calculate heading error
            heading_error = normalize_angle(yaw - self.recovery_target_heading)
            
            # If we're close enough to the target heading, transition back to NORMAL state
            if abs(heading_error) < math.radians(0):  # 5 degree threshold
                self.get_logger().info('Recovery complete: Reached target heading')
                self.state = "NORMAL"
                self.is_recovering = False

    def scan_callback(self, msg):
        # Check if there's a recent user input before processing obstacles
        current_time = self.get_clock().now()
        time_since_last_input = (current_time - self.last_input_time).nanoseconds / 1e9

        if time_since_last_input > self.input_timeout:
            # Reset user input flag if no input for more than timeout period
            self.has_user_input = False

        # Process obstacles for both normal operation and recovery
        previous_obstacle_count = len(self.current_obstacles)
        self.current_obstacles = []
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        center_angle = math.pi / 2
        angle_threshold = self.angle_threshold
        
        # Flag to detect close obstacles specifically during recovery
        close_obstacle_detected = False
        close_distance_threshold = 0.4  # 40cm threshold for close obstacles

        for i, r in enumerate(msg.ranges):
            if r == float('inf'):
                continue
            angle = angle_min + i * angle_increment
            if (center_angle - angle_threshold) <= angle <= (center_angle + angle_threshold) and r < self.obstacle_threshold:
                x_lidar = r * math.cos(angle)
                y_lidar = r * math.sin(angle)
                x_robot = y_lidar
                y_robot = -x_lidar
                self.current_obstacles.append((x_robot, y_robot))
                
                # Check for close obstacles during recovery
                if self.state == "RECOVERING" and r < close_distance_threshold:
                    close_obstacle_detected = True
                    
        # Update DWA active flag
        has_obstacles = len(self.current_obstacles) > 0
        self.dwa_active = has_obstacles
        
        # Safety interrupt during recovery
        if self.state == "RECOVERING" and close_obstacle_detected:
            self.get_logger().info('Safety interrupt: Close obstacle detected during recovery!')
            self.state = "NORMAL"  # Return to normal state
            self.is_recovering = False
            return
        
        # State transitions based on obstacle detection
        if self.state == "NORMAL" and has_obstacles:
            # Transition to AVOIDING state
            self.state = "AVOIDING"
            self.initial_heading = self.robot_pose['theta']
            self.max_heading_deviation = 0.0
            self.obstacle_detected_time = current_time
            self.original_user_cmd = self.user_cmd  # Store original command
            self.get_logger().info('Obstacle detected: Starting avoidance')
            
        elif self.state == "AVOIDING" and not has_obstacles:
            # Transition to RECOVERING state
            self.state = "RECOVERING"
            self.obstacle_cleared_time = current_time
            avoidance_duration = (self.obstacle_cleared_time - self.obstacle_detected_time).nanoseconds / 1e9
            self.recovery_target_heading = self.initial_heading  # Target heading is the initial heading
            self.is_recovering = True
            
            self.get_logger().info(f'Obstacle cleared after {avoidance_duration:.2f} seconds')
            self.get_logger().info(f'Maximum heading deviation: {math.degrees(self.max_heading_deviation):.1f} degrees')
            self.get_logger().info(f'Starting heading recovery to {math.degrees(self.recovery_target_heading):.1f} degrees')

    
    
    def cmd_vel_callback(self, msg):
        # Update the last input time and set has_user_input flag
        self.last_input_time = self.get_clock().now()
        self.has_user_input = True

        # Check if the input is actually a command to move
        if abs(msg.linear.x) > 1e-3 or abs(msg.angular.z) > 1e-3:
            # Only update user command if we're not in recovery mode
            if not self.is_recovering:
                self.user_cmd = msg
                
                # Generate user trajectory
                self.user_trajectory = self.simulate_user_trajectory(
                    msg.linear.x,
                    msg.angular.z
                )
                
                # Publish user trajectory as path
                self.publish_user_trajectory()
        else:
            # If input is essentially zero, reset the input flag
            self.has_user_input = False
            self.user_trajectory = []
            
            # Also reset states if we're stopped
            if self.state != "NORMAL":
                self.get_logger().info('User stopped: Resetting to NORMAL state')
                self.state = "NORMAL"
                self.is_recovering = False

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

    def transform_local_trajectory_to_global(self, trajectory):
        """Transform trajectory points from base_link to global frame"""
        # If global frame is base_link or transform is not available, return local trajectory
        if self.global_frame == "base_link" or not self.transform_available:
            return trajectory
            
        try:
            # Get current transform from base_link to global frame
            transform = self.tf_buffer.lookup_transform(
                self.global_frame,
                'base_link',
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)  # Short timeout
            )
            
            # Extract transform components
            trans_x = transform.transform.translation.x
            trans_y = transform.transform.translation.y
            
            # Extract rotation as quaternion
            q = transform.transform.rotation
            # Convert quaternion to angle (simplified for mostly 2D rotation)
            rot_angle = 2 * math.atan2(q.z, q.w)
            
            # Log transform details for debugging
            self.get_logger().debug(f'Applying transform: pos=({trans_x:.3f}, {trans_y:.3f}), rot={rot_angle:.3f}rad')
            
            # Transform each point - must use different variable name for output
            global_trajectory = []
            for x, y, theta, v, w in trajectory:
                # Rotate and translate point
                global_x = x * math.cos(rot_angle) - y * math.sin(rot_angle) + trans_x
                global_y = x * math.sin(rot_angle) + y * math.cos(rot_angle) + trans_y
                global_theta = theta + rot_angle
                
                # Normalize angle
                global_theta = normalize_angle(global_theta)
                    
                global_trajectory.append((global_x, global_y, global_theta, v, w))
            
            # Print first point before and after transformation for verification
            if trajectory and global_trajectory:
                x0, y0 = trajectory[0][0], trajectory[0][1]
                gx0, gy0 = global_trajectory[0][0], global_trajectory[0][1]
                self.get_logger().debug(f'First point: local=({x0:.3f}, {y0:.3f}), global=({gx0:.3f}, {gy0:.3f})')
                
            return global_trajectory
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warning(f'Transform failed: {e}')
            return trajectory
            
    def publish_user_trajectory(self):
        """Publish the user's intended trajectory as a Path message in global frame"""
        if not self.user_trajectory:
            return
            
        # Transform trajectory to global frame
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
            
            # Convert theta to quaternion (simplified for 2D rotation around Z axis)
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = math.sin(theta / 2.0)
            pose.pose.orientation.w = math.cos(theta / 2.0)
            
            path_msg.poses.append(pose)
            
        self.pub_user_path.publish(path_msg)
            
        # Debug info
        self.get_logger().debug(f'Published user trajectory with {len(path_msg.poses)} points in {self.global_frame} frame')

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

            #   collisions check - but only for the initial simulation time
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

    def calculate_score(self, candidate_traj, v, min_clearance):
        """Score based on trajectory similarity"""
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
            total_deviation += math.hypot(cx - ux, cy - uy)

        avg_deviation = total_deviation / min_len
        trajectory_score = -avg_deviation * self.trajectory_weight
        clearance_score = min_clearance * self.clearance_weight if min_clearance != float('inf') else 0.0
        reverse_bonus = 5.0 if v < 0 else 0.0

        return trajectory_score + clearance_score + reverse_bonus

    def compute_best_velocity(self):
        best_score = -float('inf')
        best_v, best_w = self.user_cmd.linear.x, self.user_cmd.angular.z
        best_trajectory = []

        v_min, v_max, w_min, w_max = self.get_dynamic_window(
            self.current_v, self.current_w
        )

        #  velocities
        for v in np.linspace(v_min, v_max, 15):
            for w in np.linspace(w_min, w_max, 15):
                valid, trajectory, min_clearance = self.simulate_trajectory(float(v), float(w))
                if valid or v < 0:
                    score = self.calculate_score(trajectory, v, min_clearance)
                    if score > best_score:
                        best_score = score
                        best_v, best_w = float(v), float(w)
                        best_trajectory = trajectory
                        
        return (best_v, best_w)

    def compute_recovery_velocity(self):
        """Compute velocity command for recovering original heading"""
        # Calculate the heading error
        heading_error = normalize_angle(self.recovery_target_heading - self.robot_pose['theta'])
        
        # Apply proportional control to the angular velocity
        # The factor 1.0 can be tuned for smoothness
        angular_velocity = 1.0 * heading_error
        
        # Limit the angular velocity
        angular_velocity = max(-self.w_max/2, min(self.w_max/2, angular_velocity))
        
        # Use a reduced forward velocity during recovery
        linear_velocity = self.original_user_cmd.linear.x * 1.2
        
        # Safety check - if the angular error is large, reduce linear velocity further
        if abs(heading_error) > math.radians(30):
            linear_velocity *= 0.5
            
        return linear_velocity, angular_velocity

    def publish_predicted_trajectory(self, trajectory):
        """Publish the predicted robot trajectory as a Path message in global frame"""
        if not trajectory:
            return
            
        # Transform trajectory to global frame
        global_trajectory = self.transform_local_trajectory_to_global(trajectory)
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
            
            # Convert theta to quaternion (simplified for 2D rotation around Z axis)
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = math.sin(theta / 2.0)
            pose.pose.orientation.w = math.cos(theta / 2.0)
            
            path_msg.poses.append(pose)
            
        self.pub_predicted_path.publish(path_msg)
                
        # Debug info
        self.get_logger().debug(f'Published predicted trajectory with {len(path_msg.poses)} points in {self.global_frame} frame')

    def publish_cmd_vel(self):
        # publish logic
        cmd = Twist()
        
        if self.state == "AVOIDING" and self.dwa_active and self.has_user_input:
            # DWA obstacle avoidance
            v, w = self.compute_best_velocity()
            cmd.linear.x = float(v)
            cmd.angular.z = float(w*0.8)  # Reduce angular velocity a bit for smoother motion
            self.current_v = cmd.linear.x
            self.current_w = cmd.angular.z
            
            # Generate and publish the extended predicted trajectory
            _, trajectory, _ = self.simulate_trajectory(cmd.linear.x, cmd.angular.z)
            if trajectory:
                # Extend the trajectory for visualization
                extended_trajectory = self.extend_trajectory(
                    trajectory,
                    self.extended_sim_time - self.short_sim_time,
                    cmd.linear.x,
                    cmd.angular.z
                )
                self.publish_predicted_trajectory(extended_trajectory)
                
        elif self.state == "RECOVERING" and self.has_user_input:
            # Heading recovery mode
            v, w = self.compute_recovery_velocity()
            cmd.linear.x = float(v)
            cmd.angular.z = float(w)
            self.current_v = cmd.linear.x
            self.current_w = cmd.angular.z
            
            # Generate and publish the extended predicted trajectory for recovery
            _, trajectory, _ = self.simulate_trajectory(cmd.linear.x, cmd.angular.z)
            if trajectory:
                # Extend the trajectory for visualization
                extended_trajectory = self.extend_trajectory(
                    trajectory,
                    self.extended_sim_time - self.short_sim_time,
                    cmd.linear.x,
                    cmd.angular.z
                )
                self.publish_predicted_trajectory(extended_trajectory)
            
        elif self.has_user_input:
            # Normal mode - use user command
            cmd = self.user_cmd
            self.current_v = cmd.linear.x
            self.current_w = cmd.angular.z
            
            # When using user command, also simulate and publish the extended trajectory
            _, trajectory, _ = self.simulate_trajectory(cmd.linear.x, cmd.angular.z)
            if trajectory:
                # Extend the trajectory for visualization
                extended_trajectory = self.extend_trajectory(
                    trajectory,
                    self.extended_sim_time - self.short_sim_time,
                    cmd.linear.x,
                    cmd.angular.z
                )
                self.publish_predicted_trajectory(extended_trajectory)
            
        self.pub_cmd_vel.publish(cmd)
        
    def extend_trajectory(self, trajectory, additional_time, v, w):
        """Extend a trajectory by simulating more steps with constant velocity"""
        if not trajectory:
            return trajectory
            
        # Start from the last point of the existing trajectory
        last_point = trajectory[-1]
        x, y, theta = last_point[0], last_point[1], last_point[2]
        
        # Calculate how many more steps to add
        additional_steps = int(additional_time / self.dt)
        extended_trajectory = trajectory.copy()  # Start with the original trajectory
        
        # Continue the simulation with constant velocity
        for _ in range(additional_steps):
            theta += w * self.dt
            x += v * math.cos(theta) * self.dt
            y += v * math.sin(theta) * self.dt
            extended_trajectory.append((x, y, theta, v, w))
            
        return extended_trajectory

def normalize_angle(angle):
    """Normalize angle to be within [-pi, pi]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle <= -math.pi:
        angle += 2 * math.pi
    return angle

def main(args=None):
    rclpy.init(args=args)
    node = DWAController()
    node.create_timer(0.1, node.publish_cmd_vel)
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()



