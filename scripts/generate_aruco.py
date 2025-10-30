from xlib.algo.cv.detector import generate_single_aruco
import cv2

if __name__ == "__main__":
    generate_single_aruco(
        aruco_type=cv2.aruco.DICT_6X6_100,
        marker_id=25,
        pixels=500,
        output_file="aruco-0.015-id25.png",
    )
