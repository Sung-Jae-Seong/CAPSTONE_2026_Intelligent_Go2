from logger import Logger
import rclpy

rclpy.init()

log = Logger(
    "/home/unitree/jiwon/InternNav/logs",
    image_encoding="bgr8",
    bridge_only=False,
    zenoh_config_path="/home/unitree/zenoh_bridge/udp_bridge.json5",
    zenoh_working_dir="/home/unitree/zenoh_bridge",
    auto_start_bridge=True,
)

# bridge_only=False 일 때만 실제 저장 subscriber가 생성됨
# log.set_ros2_topic("/camera/color/image_raw", "sensor_msgs/msg/Image", hz=1.0, raw=True)
# log.set_ros2_topic("/utlidar/robot_odom", "nav_msgs/msg/Odometry", hz=10.0)

log.add_publisher(
    pub_type="compressed",
    input_topic="/camera/color/image_raw",
    output_topic="/camera/color/image_raw/compressed",
    hz=5.0,
    jpeg_quality=40,
)

# log.add_publisher(
#     pub_type="dummy",
#     topic_name="/utlidar/robot_odom",
#     topic_type_str="nav_msgs/msg/Odometry",
# )

log.logging_start()
rclpy.spin(log)
