import numpy
import cv2
import os
from hashlib import sha256

from src.configurations import CACHE_FOLDER


def batch_knn(video_features, ads_features, k=5, use_cache=True):
    """
    Calculates KNN of each video frame against every ad frame.
    
    :param video_features: Features of each frame of the video (shape: [sampled_frames, features])
    :param ads_features: Features of each frame, of each ad (shape: [ad_no, sampled_frames, features])
    :return: KNN of every frame of the video (shape: [video_frame, i_nearest_neighbor]) 
    """
    def distance(vec1, vec2):
        return numpy.linalg.norm(numpy.asarray(vec1)-numpy.asarray(vec2))

    cache_filename = sha256(
        sha256(str(video_features).encode()).hexdigest().encode() +
        sha256(str(ads_features).encode()).hexdigest().encode()
    ).hexdigest()
    cache_path = str(CACHE_FOLDER / "knn" / cache_filename) + ".npy"
    if use_cache:
        # check if features are cached
        try:
            features = numpy.load(cache_path)
            print("info: loading knn results from cache")
            return features
        except FileNotFoundError:
            pass

    knn_result = []
    for frame_features in video_features:
        neighbors = []
        for ad_idx, ad_frames in enumerate(ads_features):
            for ad_frame_idx, ad_frame_features in enumerate(ad_frames):
                neighbors.append(("{}_{}".format(ad_idx, ad_frame_idx), distance(frame_features, ad_frame_features)))
        knn = sorted(neighbors, key=lambda element: element[1])[:k]
        knn_labels = []
        for neighbor in knn:
            knn_labels.append(neighbor[0])
        knn_result.append(knn_labels)
    print("info: knn calculated successfully")

    try:
        # save results to cache
        numpy.save(cache_path, knn_result)
    except IOError as e:
        print("warning: error saving cached knn to {}; stacktrace: {}".format(cache_path, e))
        pass
    return knn_result

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