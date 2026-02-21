import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('composite_bki_ros')
    config_file = os.path.join(pkg_dir, 'config', 'params.yaml')

    return LaunchDescription([
        Node(
            package='composite_bki_ros',
            executable='simple_node',
            name='simple_node',
            output='screen',
            parameters=[config_file]
        )
    ])
