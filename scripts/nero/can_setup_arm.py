import time
from platform import system
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

def setup_arm_can(channel: str):

    cfg = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=NeroFW.DEFAULT,
        interface="socketcan",
        channel=channel,
    )

    robot = AgxArmFactory.create_arm(cfg)

    robot.connect()

    while not robot.enable():
        robot.set_normal_mode()
        time.sleep(0.01)

    # robot.clear_joint_error()
    # time.sleep(0.01)
    # robot.reset()

    print("CAN mode enabled for " + channel)

setup_arm_can("can0")
setup_arm_can("can1")
