import time
from platform import system
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

cfg = create_agx_arm_config(
    robot=ArmModel.NERO,
    firmeware_version=NeroFW.DEFAULT,
    interface="socketcan",
    channel="can1",
)

robot = AgxArmFactory.create_arm(cfg)

robot.connect()

while not robot.enable():
    robot.set_normal_mode()
    time.sleep(0.01)

print("CAN mode enabled")

robot.set_leader_mode()

print("leader mode enabled")