import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class JoystickController(Node):
    def __init__(self):
        super().__init__('joystick_controller')
        self.get_logger().info('Joystick controller node STARTED')
        
        self.subscription = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel_usr', 10)
        
        # Configuration parameters
        self.linear_axis = 1    # Left stick 
        self.angular_axis = 2    # Right stick 
        self.deadzone = 0.15     #  deadzone
        self.max_linear = 0.8
        self.max_angular = 1.6

        self.get_logger().info(f'Configuration: Linear axis={self.linear_axis}, '
                              f'Angular axis={self.angular_axis}, '
                              f'Deadzone={self.deadzone}, '
                              f'Max linear={self.max_linear}m/s, '
                              f'Max angular={self.max_angular}rad/s')

    def joy_callback(self, msg):
        twist = Twist()
        
        #  linear velocity (
        lin_raw = msg.axes[self.linear_axis]
        twist.linear.x = self.scale_axis(lin_raw, self.max_linear)
        
        # Process angular velocity 
        ang_raw = msg.axes[self.angular_axis]
        twist.angular.z = self.scale_axis(ang_raw, self.max_angular)
        
        #  publishing
        self.publisher.publish(twist)
        self.get_logger().info(f'Published: lin={twist.linear.x:.2f}, ang={twist.angular.z:.2f}')

    def scale_axis(self, value, max_speed):
        """ deadzone and scale input to output speed"""
        if abs(value) < self.deadzone:
            return 0.0
            
        # Scale from [deadzone..1] to [0..1]
        scaled = (abs(value) - self.deadzone) / (1 - self.deadzone)
        scaled = min(max(scaled, 0.0), 1.0)  # Clamp to valid range
        return scaled * max_speed * (1 if value > 0 else -1)

def main(args=None):
    rclpy.init(args=args)
    controller = JoystickController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
