"""Offline ACT forward/backward/inference check on a local joint LeRobot dataset.

Run in act, from the project root:
  python scripts/check_lerobot_act.py data/lerobot/nero_l20_act
No robot/teleoperator SDK is imported and no hardware is connected.
"""

import argparse
import json
from pathlib import Path

import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    torch.set_num_threads(2)
    identity = json.loads((args.root / "meta/recording.json").read_text())
    dataset = LeRobotDataset(repo_id=identity["repo_id"], root=args.root)
    features = dataset_to_policy_features(dataset.features)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = ACTConfig(
        input_features={k: v for k, v in features.items() if v.type is not FeatureType.ACTION},
        output_features={k: v for k, v in features.items() if v.type is FeatureType.ACTION},
        chunk_size=60, n_action_steps=20, pretrained_backbone_weights=None, device=device,
    )
    joint_dim = len(identity["joint_names"])
    assert joint_dim in (23, 46)
    assert config.input_features["observation.state"].shape == (joint_dim,)
    assert config.output_features["action"].shape == (joint_dim,)
    dataset = LeRobotDataset(repo_id=identity["repo_id"], root=args.root,
                             delta_timestamps={"action": [i / dataset.fps for i in range(60)]})
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)))
    preprocessor, _ = make_act_pre_post_processors(config, dataset.meta.stats)
    batch = preprocessor(batch)
    policy = ACTPolicy(config).to(device)
    policy.train()
    loss, _ = policy(batch)
    assert torch.isfinite(loss), "Non-finite training loss"
    loss.backward()
    assert any(p.grad is not None for p in policy.parameters())
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
    policy.eval()
    with torch.no_grad():
        predicted = policy.predict_action_chunk(batch)
    assert predicted.shape == (1, 60, joint_dim)
    assert torch.isfinite(predicted).all()
    print(f"PASS: ACT forward/backward/inference; loss={loss.item():.6f}; "
          f"prediction={tuple(predicted.shape)}; device={device}")


if __name__ == "__main__":
    main()
