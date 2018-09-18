import unittest

from src import feature_extraction


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

if __name__ == '__main__':
    unittest.main()