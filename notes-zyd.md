按 R 开始跟随，B 开始录制，S 保存，Q 保存并退出。重复按 B 现在不会清空正在录制的数据。
python3 -m teleop.cli \
  --config configs/quest3_nero_l20.yaml \
  run --control both --execute \
  --record-dir data/episodes
## 保存结构
data/episodes/
└── episode_<UTC>/
    ├── trajectory.npz
    ├── metadata.json
    └── camera/
        ├── color.mp4
        └── depth.h5