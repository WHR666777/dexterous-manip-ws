#!/bin/bash

sudo ip link set can_arm_a down
# sudo ip link set can_arm_b down
sudo ip link set can_hand_a down
# sudo ip link set can_hand_b down

sudo ip link set can_arm_a up type can bitrate 1000000
# sudo ip link set can_arm_b up type can bitrate 1000000
sudo ip link set can_hand_a up type can bitrate 1000000
# sudo ip link set can_hand_b up type can bitrate 1000000

# python3 can_setup_arm.py
