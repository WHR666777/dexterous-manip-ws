### VR设备
左下角资源库 -> robo -> 勾选手部追踪、reconnect

### 终端~/ZJX/Nero_L20/AnyDexRetarget/example
如果使用USB连接VR设备和电脑，开启adb：
```
adb reverse tcp:63901 tcp:63901
adb reverse --list
```
打开中转转发：
```
python input/pico4_daemon.py
```
运行遥操代码：
```
python teleop_real.py --input pico4 --robot linker_l20 --hand right \
  --l20-transport can --l20-can-channel can0 \
  --l20-can-speed 100 --l20-can-command-hz 50 \
  --l20-can-max-step 5 --l20-can-max-velocity 200 \
  --l20-can-max-acceleration 1600 \
  --l20-can-deadband 1 \
  --l20-can-input-timeout 0.5
```

