import cv2
import numpy as np
import argparse


class DepthToNormalMap:
    """A class for converting a depth map image to a normal map image.


    Attributes:
        depth_map (ndarray): A numpy array representing the depth map image.
        max_depth (int): The maximum depth value in the depth map image.
    """

    def __init__(self, depth_map_path: str, max_depth: int = 255) -> None:
        """Constructs a DepthToNormalMap object.

        Args:
            depth_map_path (str): The path to the depth map image file.
            max_depth (int, optional): The maximum depth value in the depth map image.
                Defaults to 255.

        Raises:
            ValueError: If the depth map image file cannot be read.

        """
        self.depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)

        if self.depth_map is None:
            raise ValueError(
                f"Could not read the depth map image file at {depth_map_path}"
            )
        self.max_depth = max_depth

    def convert(self, output_path: str) -> None:
        """Converts the depth map image to a normal map image.

        Args:
            output_path (str): The path to save the normal map image file.

        """
        rows, cols = self.depth_map.shape

        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        # Calculate the partial derivatives of depth with respect to x and y
        dx = cv2.Sobel(self.depth_map, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(self.depth_map, cv2.CV_32F, 0, 1)

        # Compute the normal vector for each pixel
        normal = np.dstack((-dx, -dy, np.ones((rows, cols))))
        norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
        normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)

        # Map the normal vectors to the [0, 255] range and convert to uint8
        normal = (normal + 1) * 127.5
        normal = normal.clip(0, 255).astype(np.uint8)

        # Save the normal map to a file
        normal_bgr = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, normal_bgr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert depth map to normal map")
    parser.add_argument("--input", type=str, help="Path to depth map image")
    parser.add_argument(
        "--max_depth", type=int, default=255, help="Maximum depth value (default: 255)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="normal_map.png",
        help="Output path for normal map image (default: normal_map.png)",
    )
    args = parser.parse_args()

    converter = DepthToNormalMap(args.input, max_depth=args.max_depth)
    converter.convert(args.output_path)
