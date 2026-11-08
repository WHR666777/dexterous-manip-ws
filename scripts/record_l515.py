"""Record L515 color video to MP4; press Ctrl+C to stop and save.

Usage: python scripts/record_l515.py [output.mp4]
Requires: pyrealsense2, opencv-python, numpy (project's vision extra).
"""

import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", nargs="?", default=f"l515_{datetime.now():%Y%m%d_%H%M%S}.mp4",
        help="output video path (default: l515_TIMESTAMP.mp4)",
    )
    args = parser.parse_args()

    import cv2
    import numpy as np
    import pyrealsense2 as rs

    config = rs.config()
    for device in rs.context().query_devices():
        if "L515" in device.get_info(rs.camera_info.name):
            config.enable_device(device.get_info(rs.camera_info.serial_number))
            break
    else:
        raise RuntimeError("No L515 camera found.")
    width, height, fps = 960, 540, 30
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    pipeline = rs.pipeline()
    pipeline.start(config)
    writer = None
    try:
        writer = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open output video: {args.output}")
        print(f"Recording to {args.output}. Press Ctrl+C to stop.", flush=True)
        while True:
            color = pipeline.wait_for_frames().get_color_frame()
            if color:
                writer.write(np.asanyarray(color.get_data()))
    except KeyboardInterrupt:
        print(f"\nSaved: {args.output}")
    finally:
        if writer is not None:
            writer.release()
        pipeline.stop()


if __name__ == "__main__":
    main()
