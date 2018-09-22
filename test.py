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
            "ads"
        )
        self.assertTrue(features.any())


class TestKNN(unittest.TestCase):
    def test_knn_from_features(self):
        ad1 = [[4, 4], [5, 5]]
        ad2 = [[8, 8], [9, 9], [10, 10]]
        video = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]]
        video_ftrs = video
        ads_ftrs = [ad2, ad1]
        knn = video_tools.batch_knn(video_ftrs, ads_ftrs, use_cache=False)
        # 1st ad
        self.assertEqual(knn[3][0], '1_0')
        self.assertEqual(knn[4][0], '1_1')

        # 2nd ad
        self.assertEqual(knn[7][0], '0_0')
        self.assertEqual(knn[8][0], '0_1')
        self.assertEqual(knn[9][0], '0_2')


if __name__ == '__main__':
    unittest.main()
