import cv2
import numpy as np
import pyrealsense2 as rs

pipeline, config = rs.pipeline(), rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
try:
    device = profile.get_device()
    usb_info = rs.camera_info.usb_type_descriptor
    usb_type = device.get_info(usb_info) if device.supports(usb_info) else "未知"
    print("Camera:", device.get_info(rs.camera_info.name))
    print("USB connection:", usb_type, flush=True)
    print("Color stream:", profile.get_stream(rs.stream.color))
    while True:
        color = pipeline.wait_for_frames().get_color_frame()
        if color:
            cv2.imshow("L515 - press q to quit", np.asanyarray(color.get_data()))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
