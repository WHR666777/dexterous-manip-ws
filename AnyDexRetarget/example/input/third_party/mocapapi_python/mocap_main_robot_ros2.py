from mocap_main_base import MCPBase
from mocap_api import MCPAvatar, MCPRobot
import asyncio

import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class MCPMainRobotROS2(MCPBase):
    def __init__(self):
        super().__init__()
        rclpy.init()
        self.node = Node("json_to_joint_state_publisher")
        self.publisher = self.node.create_publisher(JointState, "/joint_states", 10)
        json_file_path = './retarget.json'
        self.robot = MCPRobot(open(json_file_path).read())
        # Load joint names from JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        self.joint_names = data['urdfJointNames']  

    def handleAvatar(self, avatar_handle):
        try:
            # Handle avatar update event
            avatar = MCPAvatar(avatar_handle)  # Get avatar data
            self.robot.update_robot(avatar)
            self.robot.run_robot_step()
            # print(robot.get_robot_ros_frame_json())

            # Get real-time data from the robot
            real_time_data = json.loads(self.robot.get_robot_ros_frame_json()[0])

            # Create JointState message
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = self.node.get_clock().now().to_msg()
            joint_state_msg.name = self.joint_names

            # Initialize joint positions
            joint_positions = [0.0] * len(self.joint_names)

            # Fill the joint positions with real-time data
            for i, name in enumerate(self.joint_names):                    
                if name in real_time_data['joint_positions']:
                    joint_positions[i] = real_time_data['joint_positions'][name]

            joint_state_msg.position = joint_positions
            self.publisher.publish(joint_state_msg)

            # print(f"Joint positions: {joint_positions}")

        except Exception as e:
            print(e)

    def main(self):
        asyncio.run(self.main_async())
        

if __name__ == '__main__':
    MCPMainRobotROS2().main()  # Start the main program