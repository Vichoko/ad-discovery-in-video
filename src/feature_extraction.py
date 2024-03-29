from enum import Enum
from os import listdir
import numpy

from src.configurations import DATA_FOLDER, SAMPLES_PER_SECOND, SAMPLING_DIMENSIONS, SOBEL_THRESH, CACHE_FOLDER, \
    SUPPORTED_EXTENSIONS
import cv2


class FeatureType(Enum):
    GRAY_SCALE = 1
    SOBEL_GRAD_CONCAT = 2
    SOBEL_GRAD_MAGNITUDE = 4
    SOBEL_THRESH_BINARY = 3


def extract_features_from_video(filename,
                                fps=30,
                                sps=SAMPLES_PER_SECOND,
                                ft_type=FeatureType.SOBEL_THRESH_BINARY,
                                data_folder_path=DATA_FOLDER,
                                use_cache=True,
                                cache_prefix=""):
    """
    Extract features from video
    :param fps: Frames per second of the video to be sampled
    :param sps: Samples per second to be sampled
    :param ft_type: Feature type to be extracted, options are in FeatureType Enum
    :param data_folder_path: Path of the folder containing the video file to be sampled
    :param use_cache: Flag to use cache if available
    :param cache_prefix: Prefix of the cached files
    :param filename: Filename from video in DATA_FOLDER
    :return: Features of video, with shape [sampled_frames, features]
    """

    video_path = str(data_folder_path / filename)
    cache_filename = "{}{}_{}{}_{}spd_{}".format(
        cache_prefix,
        filename,
        ft_type.name,
        SOBEL_THRESH,
        sps,
        str(SAMPLING_DIMENSIONS)
    )
    cache_path = str(CACHE_FOLDER / "features" / cache_filename) + ".npy"
    if use_cache:
        # check if features are cached
        try:
            features = numpy.load(cache_path)
            print("info: loading features of {} from cache".format(filename))
            return features
        except FileNotFoundError:
            pass

    capture = cv2.VideoCapture(video_path)
    skipped_frames = 0
    delta = int(fps / sps)

    features = []

    while capture.grab():
        # sampling mechanism
        skipped_frames += 1
        if skipped_frames < delta:
            continue

        # getting frame
        retval, frame = capture.retrieve()
        if not retval:
            continue
        skipped_frames = 0

        # down-sampling dimensions
        frame = cv2.resize(frame, SAMPLING_DIMENSIONS)

        # gray scaling
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ft_type == FeatureType.GRAY_SCALE:
            # feature type is Gray Scale; append and continue
            features.append(gray_frame.flatten())
            continue

        # sobel filters
        grad_x = cv2.Sobel(gray_frame, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        grad_y = cv2.Sobel(gray_frame, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
        if ft_type == FeatureType.SOBEL_GRAD_CONCAT:
            # feature type is Gradient Concatenation; concat, append and continue
            features.append(numpy.concatenate((grad_x.flatten(), grad_x.flatten())))
            continue

        # gradient magnitude approximation
        grad_magnitude = numpy.sqrt(numpy.square(grad_x) + numpy.square(grad_y))
        if ft_type == FeatureType.SOBEL_GRAD_MAGNITUDE:
            # feature type is Gradient Magnitude; append and continue
            features.append(grad_magnitude.flatten())
            continue

        # to binary (0s and 255s)
        _, borders = cv2.threshold(grad_magnitude, thresh=SOBEL_THRESH, maxval=255, type=cv2.THRESH_BINARY)
        # feature type is Thresh; append and continue
        features.append(borders.flatten())
    features = numpy.asarray(features)
    print("info: features of {} extracted succesfully".format(filename))

    try:
        numpy.save(cache_path, features)
    except IOError as e:
        print("warning: error saving cached features to {}; stacktrace: {}".format(cache_path, e))
        pass
    return features


def extract_features_from_video_folder(foldername,
                                       video_extensions=SUPPORTED_EXTENSIONS,
                                       fps=30,
                                       sps=SAMPLES_PER_SECOND,
                                       ft_type=FeatureType.SOBEL_THRESH_BINARY):
    """
    Extract features of each video in the folder
    :param video_extensions: 
    :param fps: 
    :param sps: 
    :param ft_type: 
    :param foldername: Folder path containing ad videos in DATA_FOLDER.
    :return: Features of each video, with shape [videos_in_folder, sampled_frames, features]
    """
    features = []
    video_names = []
    video_folder_path = DATA_FOLDER / foldername
    for filename in listdir(str(video_folder_path)):
        if filename[-3:] not in video_extensions:
            # file in folder isnt video filetype
            continue
        features.append(extract_features_from_video(filename,
                                                    fps,
                                                    sps,
                                                    ft_type,
                                                    video_folder_path,
                                                    cache_prefix=foldername+"_"))
        video_names.append(filename.split(".")[0])
    if not features:
        raise AssertionError("Feature vector is empty. Check extensions () and if video folder is empty.".format(
            str(video_extensions)
        ))
    return numpy.asarray(features), video_names
