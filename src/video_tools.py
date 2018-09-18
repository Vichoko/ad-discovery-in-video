import numpy
import cv2
import os


def batch_knn(video_features, ads_features):
    """
    Calculates KNN of each video frame against every ad frame.
    
    :param video_features: Features of each frame of the video.
    :param ads_features: Features of each frame, of each ad.
    :return: KNN of every frame of the video (shape: [video_frame, i_nearest_neighbor]) 
    """
    pass


def ads_detector(knn_list):
    """
    Identify ad appearances in the video based on KNN results.
    Export results to APPEARANCES_OUTFILE.
    :param knn_list: KNN of each frame, 
    :return: None
    """
    pass

"""
From https://github.com/juanbarrios/teaching-MIR/blob/master/python/Ejemplos.ipynb
"""

def mostrar_frame(window_name, imagen, valorAbsoluto = False, escalarMin0Max255 = False):
    if valorAbsoluto:
        imagen_abs = numpy.abs(imagen)
    else:
        imagen_abs = imagen
    if escalarMin0Max255:
        imagen_norm = cv2.normalize(imagen_abs, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    else:
        imagen_norm = imagen_abs
    cv2.imshow(window_name, imagen_norm)