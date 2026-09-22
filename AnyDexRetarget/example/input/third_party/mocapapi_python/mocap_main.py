from mocap_main_base import MCPBase
from mocap_api import MCPAvatar
import asyncio

class MCPMain(MCPBase):
    def __init__(self):
        super().__init__()

    def handleAvatar(self, avatar_handle):
        # Handle avatar update event
        avatar = MCPAvatar(avatar_handle)  # Get avatar data
        joints = avatar.get_joints()  # Get all joint data
        str_data = '{'
        for joint in joints:
            link_name = joint.get_name()  # Get joint name
            position = joint.get_local_position()  # Get joint position
            rotation = joint.get_local_rotation()  # Get joint rotation
            str_data += f'{link_name} : {position}, {rotation}'
        str_data += '}'
        print(f"links_data: {str_data}")  # Print joint data

    def main(self):
        asyncio.run(self.main_async())

if __name__ == '__main__':
    MCPMain().main()