import unittest
import cv2
import numpy as np
import os
from depth_to_normal_map import DepthToNormalMap


class TestDepthToNormalMap(unittest.TestCase):
    def setUp(self):
        self.depth_map_path = "test_depth_map.png"
        self.normal_map_path = "test_normal_map.png"
        self.max_depth = 255
        self.expected_shape = (480, 640, 3)

        # create a sample depth map image
        self.depth_map = np.random.randint(
            low=0, high=self.max_depth, size=self.expected_shape[:2], dtype=np.uint16
        )
        cv2.imwrite(self.depth_map_path, self.depth_map)

    def test_converting_depth_map_to_normal_map(self):
        # create an instance of DepthToNormalMap
        converter = DepthToNormalMap(self.depth_map_path, max_depth=self.max_depth)

        # convert the depth map to normal map
        converter.convert(self.normal_map_path)

        # assert the output file exists
        self.assertTrue(os.path.isfile(self.normal_map_path))

        # read the output normal map
        normal_map = cv2.imread(self.normal_map_path)

        # assert the shape and data type of the normal map
        self.assertEqual(normal_map.shape, self.expected_shape)
        self.assertEqual(normal_map.dtype, np.uint8)

    def tearDown(self):
        # remove the test depth and normal map images
        if os.path.isfile(self.depth_map_path):
            os.remove(self.depth_map_path)

        if os.path.isfile(self.normal_map_path):
            os.remove(self.normal_map_path)


if __name__ == "__main__":
    unittest.main()
