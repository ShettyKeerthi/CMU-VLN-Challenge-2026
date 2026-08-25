from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="vln_ai_module_nano",
            executable="vln_ai_module_nano_node",
            name="vln_ai_module_nano_node",
            output="screen",
        ),
    ])
