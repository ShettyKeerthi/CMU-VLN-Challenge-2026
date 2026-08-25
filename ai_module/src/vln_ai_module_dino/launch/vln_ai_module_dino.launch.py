from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="vln_ai_module_dino",
            executable="vln_ai_module_dino_node",
            name="vln_ai_module_dino_node",
            output="screen",
        ),
    ])
