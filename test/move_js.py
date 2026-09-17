import time
import numpy as np
from pyAgxArm import (
    create_agx_arm_config,
    AgxArmFactory,
    NeroFW,
)

cfg = create_agx_arm_config(
    robot="nero",
    comm="can",
    channel="can0",
    firmeware_version=NeroFW.V111,
)

robot = AgxArmFactory.create_arm(cfg)
robot.connect()

while not robot.enable():
    time.sleep(0.01)

time.sleep(0.2)

def get_q():
    q = []
    for i in range(1, 8):
        msg = robot.get_motor_states(i)
        q.append(float(msg.msg.position))
    return np.asarray(q)

q = get_q()

print("当前(deg):")
print(np.rad2deg(q))

target = q.copy()

# 只动 J1 0.2°
target[0] += np.deg2rad(0.2)
target[6] += np.deg2rad(20)

print("目标(deg):")
print(np.rad2deg(target))

input("确认安全后输入 MOVE：")
print(target.tolist())

robot.move_js(target.tolist())

q0 = get_q()

for _ in range(20):
    time.sleep(0.1)
    now = get_q()

    print(
        "运动量(deg)=",
        np.round(np.rad2deg(now - q0), 4),
        "目标误差(deg)=",
        np.round(np.rad2deg(target - now), 4),
    )