import cv2
from xlib.device.sensor.camera import RealSenseCamera

third_view_camera_serial = "408122071109"
first_vew_camera_serial = "f1421776"


first_view_camera = RealSenseCamera(exposure_time=300, serial_number=third_view_camera_serial)
# third_view_camera = RealSenseCamera(exposure_time=300, serial_number=third_view_camera_serial)

while True:
    first_view_color, _ = first_view_camera.get_frame()
    # third_view_color, _ = third_view_camera.get_frame()
    cv2.imshow("first_view_color", first_view_color)
    # cv2.imshow("third view color", third_view_color)
    cv2.waitKey(1)
    