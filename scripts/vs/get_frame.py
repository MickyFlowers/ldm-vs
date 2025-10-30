from xlib.device.sensor.camera import RealSenseCamera
import cv2
import time

camera = RealSenseCamera(exposure_time=700, serial_number="f1421776")
for i in range(30):
    camera.get_frame()
    time.sleep(1.0 / 20.0)
while True:
    color_image, depth_image = camera.get_frame()
    # print(color_image.shape)
    cv2.imshow("color_image", color_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.imwrite("data/vs_examples/vs_finetune_charge.jpg", color_image)
        cv2.destroyAllWindows()
        break
