from PIL import Image, ImageFilter
import cv2
import numpy as np


# def retinex(image, sigma=100):
#     """
#     实现单尺度Retinex算法
#     :param image: 输入图像
#     :param sigma: 高斯模糊的标准差
#     :return: 增强后的图像
#     """
#     # 将图像转换为浮点数
#     image = image.astype(np.float32)
#     # 计算图像的对数
#     log_image = np.log1p(image)
#     # 对图像进行高斯模糊
#     blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)
#     # 计算模糊图像的对数
#     log_blurred_image = np.log1p(blurred_image)
#     # 计算Retinex增强后的图像
#     retinex_image = log_image - log_blurred_image
#     # 将图像转换回原始数据类型
#     retinex_image = np.expm1(retinex_image)
#     # 归一化图像
#     retinex_image = cv2.normalize(
#         retinex_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
#     )
#     return retinex_image


def canny_edge_detection(image, low_threshold=30, high_threshold=150):
    """
    实现Canny边缘检测
    :param image: 输入图像
    :param low_threshold: 低阈值
    :param high_threshold: 高阈值
    :return: 边缘检测后的图像
    """
    # 将图像转换为灰度图
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges


# # 读取图像
# image = cv2.imread("../data/good_random_data_single/seg/000-004.jpg")

# # 应用Retinex算法增强图像
# enhanced_image = retinex(image)

# # 应用Canny边缘检测
# edges = canny_edge_detection(enhanced_image)

# # 显示结果
# cv2.imshow("Original Image", image)
# cv2.imshow("Enhanced Image", enhanced_image)
# cv2.imshow("Edges", edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Open an image file
    image = cv2.imread(
        "../data/good_random_data_single/img/000-004.jpg", cv2.IMREAD_GRAYSCALE
    )
    # adaptive_threshold = cv2.adaptiveThreshold(
    #     image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    # )
    # image = cv2.medianBlur(image, 5)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # image = clahe.apply(image)
    # image = cv2.cvtColor(adaptive_threshold, cv2.COLOR_GRAY2BGR)
    # adaptive_threshold = cv2.adaptiveThreshold(
    #     image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    # )
    # image = cv2.cvtColor(adaptive_threshold, cv2.COLOR_GRAY2BGR)
    # image = cv2.Laplacian(
    # image, cv2.CV_64F, ksize=7, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT
    # )
    image = canny_edge_detection(image)
    cv2.imshow("image", image)
    cv2.waitKey(0)
