import unittest

from src import feature_extraction, video_tools


class TestFeatureExtraction(unittest.TestCase):
    def test_extract_features_single_video(self):
        features = feature_extraction.extract_features_from_video(
            "test_video.mpg"
        )
        self.assertTrue(features.any())

    def test_extract_features_video_folder(self):
        features = feature_extraction.extract_features_from_video_folder(
            "test_ads"
        )
        self.assertTrue(features.any())


class TestKNN(unittest.TestCase):

    def test_knn_from_features(self):
        ad0 = [[8, 8], [9, 9], [10, 10]]
        ad1 = [[4, 4],[5, 5]]

        video = [[1, 1], [2, 2], [4, 4], [4, 4], [5, 5], [5, 5], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]]

        video_ftrs = video
        ads_ftrs = [ad0, ad1]
        knn = video_tools.batch_knn(video_ftrs, ads_ftrs, use_cache=False)
        # manual detection
        # 1st ad
        self.assertEqual(knn[7][0], '0_0')
        self.assertEqual(knn[8][0], '0_1')
        self.assertEqual(knn[9][0], '0_2')

        # 2nd ad
        self.assertEqual(knn[3][0], '1_0')
        self.assertEqual(knn[4][0], '1_1')

        # automated detection
        detections = video_tools.ads_detector(knn, [len(ad0), len(ad1)])
        # 1st ad
        starting_frame = detections[0]['starting_frame']
        for ad_frame_idx, ad_frame in enumerate(ad0):
            self.assertEqual(video[starting_frame+ad_frame_idx], ad_frame)

        starting_frame = detections[1]['starting_frame']
        for ad_frame_idx, ad_frame in enumerate(ad1):
            self.assertEqual(video[starting_frame+ad_frame_idx], ad_frame)



if __name__ == '__main__':
    unittest.main()
