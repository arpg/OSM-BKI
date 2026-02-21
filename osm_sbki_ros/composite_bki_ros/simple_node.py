import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import sys
import os

# Try to import composite_bki_cpp
# If it's not in the path, we can try to find it relative to this file if needed,
# but usually it's better to rely on PYTHONPATH.
try:
    import composite_bki_cpp
except ImportError:
    # Fallback: try to add the parent directory's src to path if running from source
    # This assumes the structure: composite_bki/ros_example/composite_bki_ros/simple_node.py
    # and composite_bki/src/composite_bki_cpp.so exists
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up 3 levels: composite_bki_ros -> ros_example -> composite_bki
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    src_path = os.path.join(project_root, 'src')
    if os.path.exists(src_path):
        sys.path.append(src_path)
        try:
            import composite_bki_cpp
        except ImportError:
            print("Error: Could not import composite_bki_cpp. Please ensure it is in your PYTHONPATH.")
            sys.exit(1)
    else:
        print("Error: Could not import composite_bki_cpp. Please ensure it is in your PYTHONPATH.")
        sys.exit(1)

class SimpleBKINode(Node):
    def __init__(self):
        super().__init__('simple_node')

        # Parameters
        self.declare_parameter('osm_path', '')
        self.declare_parameter('config_path', '')
        self.declare_parameter('resolution', 0.1)
        self.declare_parameter('input_topic', '/velodyne_points')
        self.declare_parameter('output_topic', '/bki/semantic_cloud')
        self.declare_parameter('update_map', False)

        osm_path = self.get_parameter('osm_path').value
        config_path = self.get_parameter('config_path').value
        res = self.get_parameter('resolution').value
        
        if not osm_path or not config_path:
            self.get_logger().error("OSM path and Config path are required!")
            # We can't exit in __init__ easily, but we can prevent further execution
            return

        # Initialize BKI
        self.get_logger().info(f"Initializing BKI with OSM: {osm_path}")
        try:
            self.bki = composite_bki_cpp.PyContinuousBKI(
                osm_path=osm_path,
                config_path=config_path,
                resolution=res,
                use_semantic_kernel=True,
                use_spatial_kernel=True
            )
        except Exception as e:
            self.get_logger().error(f"Failed to initialize BKI: {e}")
            return

        # Subscribers & Publishers
        in_topic = self.get_parameter('input_topic').value
        out_topic = self.get_parameter('output_topic').value
        
        self.sub = self.create_subscription(
            PointCloud2, in_topic, self.callback, 10)
        self.pub = self.create_publisher(
            PointCloud2, out_topic, 10)
            
        self.get_logger().info("Node started.")

    def callback(self, msg):
        if not hasattr(self, 'bki'):
            return

        # 1. Convert ROS -> Numpy
        # Read x, y, z. If updating map, you might need 'label' or 'intensity' too.
        # We use a generator to save memory, then convert to numpy
        field_names = ("x", "y", "z")
        try:
            points_gen = pc2.read_points(msg, field_names=field_names, skip_nans=True)
            points = np.array(list(points_gen), dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"Failed to read points: {e}")
            return
        
        if points.size == 0:
            return

        # 2. BKI Operation
        # (Optional) Update map if enabled
        if self.get_parameter('update_map').value:
            # Placeholder: requires labels from somewhere
            # If you had labels, you would call:
            # self.bki.update(labels, points)
            pass 
            
        # Inference
        try:
            pred_labels = self.bki.infer(points)
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return

        # 3. Convert Numpy -> ROS
        # We create a structured array for PointCloud2
        # Fields: x, y, z, intensity (used for label visualization)
        # Note: 'intensity' is a float, so we cast labels to float. 
        # You could use a custom field 'label' (uint32) but rviz visualizes intensity easily.
        dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32)]
        out_points = np.zeros(points.shape[0], dtype=dtype)
        out_points['x'] = points[:, 0]
        out_points['y'] = points[:, 1]
        out_points['z'] = points[:, 2]
        out_points['intensity'] = pred_labels.astype(np.float32)

        out_msg = pc2.create_cloud(msg.header, dtype, out_points)
        self.pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleBKINode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
