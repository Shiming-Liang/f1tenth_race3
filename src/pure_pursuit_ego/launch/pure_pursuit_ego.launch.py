from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch the pure pursuit node
    pure_pursuit_ego_node = Node(
        package='pure_pursuit_ego',
        executable='pure_pursuit_node'
    )

    return LaunchDescription([
        pure_pursuit_ego_node
    ])