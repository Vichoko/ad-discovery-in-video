import numpy

from hashlib import sha256

from src.configurations import CACHE_FOLDER, SAMPLES_PER_SECOND, APPEARANCES_OUTFILE, SCORE_THRESHOLD

DEBUG = True
if DEBUG:
    all_scores = []
    passed_scores = []
    all_scores_mean_n_std = lambda: (numpy.mean(all_scores), numpy.std(all_scores))
    passed_scores_mean_n_std = lambda: (numpy.mean(passed_scores), numpy.std(passed_scores))


def batch_knn(video_features, ads_features, k=5, use_cache=True):
    """
    Calculates KNN of each video frame against every ad frame.
    
    :param k: k of the KNN
    :param use_cache: Flag to use cached version if available
    :param video_features: Features of each frame of the video (shape: [sampled_frames, features])
    :param ads_features: Features of each frame, of each ad (shape: [ad_no, sampled_frames, features])
    :return: KNN of every frame of the video (shape: [video_frame, i_nearest_neighbor]) 
    """

    def distance(vec1, vec2):
        return numpy.linalg.norm(numpy.asarray(vec1) - numpy.asarray(vec2))

    cache_filename = sha256(
        sha256(str(video_features).encode()).hexdigest().encode() +
        sha256(str(ads_features).encode()).hexdigest().encode()
    ).hexdigest()
    cache_path = str(CACHE_FOLDER / "knn" / cache_filename) + str(k) + ".npy"
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


def ads_detector(knn_list, video_name, ad_lengths, ad_names):
    """
    Identify ad appearances in the video based on KNN results.
    Export results to APPEARANCES_OUTFILE.
    :param video_name: name of the video to be shown in outfile
    :param ad_names: names of ads to be shown in outfile
    :param ad_lengths: List of frames sampled for each ad in the same order as the ads_features passed to the KNN.
    :param knn_list: KNN of each frame of the original video.
    :return: None
    """
    ad_matching_list = []

    for ad_idx, ad_length_in_frames in enumerate(ad_lengths):
        # for each possible ad
        # calculate a score list that represents the probability that the ad is starting at each frame of the video
        ad_intersection_scores = []
        ad_estimated_length_per_video_frame = []

        for starting_frame_idx, _ in enumerate(knn_list):
            # for each frame of the video, check if the ad starts at this video frame
            # check the intersection of the ad's frames within the knn results
            # if intersection is 0, then knn didn't discover any ad's frame starting in this frame
            # if intersection is 1, then knn matched every ad's frame with corresponding video frame sequentially
            ad_intersection_score = 0
            intersection_frame_counter = 0

            for ad_frame_idx in range(ad_length_in_frames):
                # check if each ad frame matches sequentially within the KNN frames,
                # starting in index: frame_idx UNTIL frame_idx+ad_length
                ad_frame_name = "{}_{}".format(ad_idx, ad_frame_idx)  # KNN output format: {ad_idx}_{frame_idx}
                try:
                    knn_priority = list(knn_list[starting_frame_idx+ad_frame_idx]).index(ad_frame_name) + 1  # 1
                    # means that this ad frame is the nearest
                    # the score given decays as the matched frame is farther in the KNN result, 1 is max
                    ad_intersection_score += 1.0 / knn_priority
                    intersection_frame_counter = ad_frame_idx
                    # if ad_frame_name in knn_list[relative_frame_idx]:
                    #     ad_intersection_score += 1.0
                    #     intersection_frame_counter = ad_frame_idx
                except ValueError:
                    # ad frame doesn't appear in the frame's knn
                    pass
                except IndexError:
                    break
            probability = 1.0 * ad_intersection_score / ad_length_in_frames
            ad_intersection_scores.append(probability)
            ad_estimated_length_per_video_frame.append(intersection_frame_counter)

            if DEBUG:
                all_scores.append(probability)
            if float(probability) > SCORE_THRESHOLD:
                # if knn discovered a sequence composed by at least a half of the ad's frames, then
                # mark it as an occurrence
                ad_matching_list.append({'ad_idx': ad_idx,
                                         'ad_length_in_frames': intersection_frame_counter,
                                         'score': probability,
                                         'starting_frame': starting_frame_idx,
                                         })
                if DEBUG:
                    passed_scores.append(probability)
    # finally,
    # export to file
    ad_matching_list = sorted(ad_matching_list, key=lambda dic: dic['starting_frame'])
    with open(APPEARANCES_OUTFILE, 'w') as fp:
        for ad_detected in ad_matching_list:
            fp.write('{}\t{}\t{}\t{}\n'.format(
                video_name,
                frame_idx_to_seconds(ad_detected['starting_frame']),
                frame_idx_to_seconds(ad_detected['ad_length_in_frames']),
                ad_names[ad_detected['ad_idx']],
                #                ad_detected['score']
            ))
        print("info: Detected ads exported to {}".format(APPEARANCES_OUTFILE))
    if DEBUG:
        print("debug: score mean {}, std {}".format(all_scores_mean_n_std()[0], all_scores_mean_n_std()[1]))
        print("debug: passed score mean {}, std {}".format(
            passed_scores_mean_n_std()[0],
            passed_scores_mean_n_std()[1])
        )
    return ad_matching_list


def frame_idx_to_seconds(frame_idx, sps=SAMPLES_PER_SECOND):
    """
    Transform a frame_id to the timestamp in seconds of the video, based on the sampling configuration.
    :param frame_idx: Frame idx to be timestamped
    :param sps: Samples per second used in sampling of the features
    :return: 
    """
    return 1.0 * frame_idx / sps


def get_ad_lengths_in_frames(ads_features):
    """
    Self-explainatory
    :param ads_features: has dimension [ad_idx, ad_frame_idx, feature_idx]
    :return: Ad length in frames
    """
    ad_lengths_in_frames = []
    for ad_features in ads_features:
        ad_lengths_in_frames.append(ad_features.shape[0])
    return ad_lengths_in_frames

