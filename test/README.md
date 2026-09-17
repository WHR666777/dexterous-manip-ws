# 独立 CAN 示例

在项目根目录、nero 环境运行；这些是手动硬件脚本，不是 pytest 测试。
原有根目录 test.py 和 tests/ 不受影响。

```bash
conda activate nero
python test/get_flange.py
python test/get_tcp.py
python test/get_joints.py
python test/get_status.py
python test/get_firmware.py
python test/reset.py
python test/move_pose.py
python test/move_js.py
```

- `get_flange.py`：法兰位姿，m/rad。正确拼写是 flange。
- `get_tcp.py`：TCP 位姿 `[x,y,z,roll,pitch,yaw]`，m/rad。默认零 TCP offset，与法兰重合；其他进程的 offset 不会自动继承。
- `get_joints.py`：七轴角度，rad。
- `get_status.py`：控制器状态；0正常、1急停、2无解。
- 上述四个读取脚本约5秒结束；首次反馈最多等待5秒。不主动使能或开 CAN push。
- `get_firmware.py`：发送查询并打印实际固件版本。
- `reset.py`：输入 RESET 后执行电子急停→复位→正常模式→失能→重新使能。失能/使能各限时5秒，最后连续三次确认正常、CAN控制、七轴使能。失能会导致失去力矩，须可靠支撑；成功后保持使能。
- `move_pose.py`：填写文件顶部 TARGET_POSE，或运行时输入六个数 `[x,y,z,roll,pitch,yaw]`；Base下法兰绝对位姿，m/rad（xyz欧拉角）。输入 MOVE 后使能，以5%速度运动，不回原位。异常/超时尝试急停。
- `move_js.py`：同样输入绝对法兰位姿，复用遥操作脚本的 NeroIK，以当前关节为种子求一次局部逆解，再以目标50Hz重复发送同一组关节目标，最多10秒。每0.1秒检查到位，每0.5秒打印误差和平均发送频率。保留每轴5°约束和模型/反馈一致性检查；无解时拒绝。JS没有轨迹规划，不使用速度百分比。不会启动SteamVR或Tracker。

通道取自 config.py，封装固定 V111。只运行一个控制脚本；读不到反馈时先核对 CAN 接口和设备推送。脚本退出断开通信，不自动失能。电子急停依赖 CAN 通信，不替代实体急停。
