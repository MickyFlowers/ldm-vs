from xlib.device.sensor.camera import RealSenseCamera
import cv2

camera = RealSenseCamera(exposure_time=800)

while True:
    color_image, depth_image = camera.get_frame()
    cv2.imshow("color_image", color_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.imwrite("data/vs_examples/vs_charge.jpg", color_image)
        
        cv2.destroyAllWindows()
        break
