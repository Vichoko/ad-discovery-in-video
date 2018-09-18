import numpy

from src.configurations import DATA_FOLDER, SAMPLES_PER_SECOND, SAMPLING_DIMENSIONS, SOBEL_THRESH
import cv2

def extract_features_from_video(filename,
                                fps=30,
                                sps=SAMPLES_PER_SECOND):
    """
    Extract features from video
    :param filename: Path of video
    :return: Features of video, with shape [sampled_frames, features]
    """
    capture = cv2.VideoCapture(str(DATA_FOLDER / filename))
    skipped_frames = 0
    delta = int(fps / sps)

    feature_type = "SOBEL_THRESH"

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

        # downsample dimensions
        frame = cv2.resize(frame, SAMPLING_DIMENSIONS)

        # gray
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # sobel filters
        grad_x = cv2.Sobel(gray_frame, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
        grad_y = cv2.Sobel(gray_frame, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

        # gradient magnitude approximation
        grad_magnitude = numpy.sqrt(numpy.square(grad_x) + numpy.square(grad_y))

        # to binary (0s and 255s)
        retval, borders = cv2.threshold(grad_magnitude, thresh=SOBEL_THRESH, maxval=255, type=cv2.THRESH_BINARY)
        features.append(borders.flatten())
    return numpy.asarray(features)

def extract_features_from_video_folder(folderpath):
    """
    Extract features of each video in the folder
    :param folderpath: Folder path containing N videos.
    :return: Features of each video, with shape [N, frames, features]
    """
    pass
