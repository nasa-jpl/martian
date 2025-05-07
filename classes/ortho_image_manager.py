import os
# os.environ["CV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,60))
import cv2

import numpy as np
import matplotlib.pyplot as plt
from utils.transform_utils import img2blender, blender2img



class OrthoImageManager:
    def __init__(self, img_path, img_ortho_size):
        self._img = cv2.imread(img_path)
        self._img_ortho_size = img_ortho_size # img ortho width, height
        self._img_size = [self.img.shape[1], self.img.shape[0]]
        self._px_res = None
        self._north = None
        self._west = None
        self._east = None
        self._south = None
        self._center = None
        self._north_blend = None
        self._west_blend = None
        self._east_blend = None
        self._south_blend = None
        self._center_blend = None
        self._rot_matrix = None
        self._terrain_ortho_size = [None, None] # terrain ortho width , height
        


    def print_data(self):
        print(f"(img_width, img_height):{(self.img_size[0] ,self.img_size[1])}")
        print(f"px_res:{self.px_res}")
        print(f"North corner: img coord.={self.north}, Blender coord.={self.north_blend}")
        print(f"West corner: img coord.={self.west}, Blender coord.={self.west_blend}")
        print(f"East corner: img coord.={self.east}, Blender coord.={self.east_blend}")
        print(f"South corner: img coord.={self.south}, Blender coord.={self.south_blend}")
        print(f"Center: img coord.={self.center}, Blender coord.={self.center_blend}")

    @property
    def img(self):
        return self._img
    
    @property
    def img_ortho_size(self):
        return self._img_ortho_size
    
    @property
    def img_size(self):
        return self._img_size
    
    @property
    def px_res(self):
        if self._px_res is None:
            self._px_res = self.img_ortho_size[0]/self.img_size[0] # img_ortho_width/img_width [m/px]
        return self._px_res

    @property
    def north(self):
        if self._north is None:
            self._north = self._find_corner('north')
        return self._north

    @property
    def west(self):
        if self._west is None:
            self._west = self._find_corner('west')
        return self._west

    @property
    def east(self):
        if self._east is None:
            self._east = self._find_corner('east')
        return self._east

    @property
    def south(self):
        if self._south is None:
            self._south = self._find_corner('south')
        return self._south

    @property
    def center(self):
        if self._center is None:
            center_x = (self.north[0] + self.east[0] + self.south[0] + self.west[0]) / 4
            center_y = (self.north[1] + self.east[1] + self.south[1] + self.west[1]) / 4
            self._center = np.array([center_x, center_y])
        return self._center

    @property
    def north_blend(self):
        if self._north_blend is None:
            self._north_blend = img2blender(self.north, self.img_size, self.px_res)
        return self._north_blend

    @property
    def west_blend(self):
        if self._west_blend is None:
            self._west_blend = img2blender(self.west, self.img_size, self.px_res)
        return self._west_blend

    @property
    def east_blend(self):
        if self._east_blend is None:
            self._east_blend = img2blender(self.east, self.img_size, self.px_res)
        return self._east_blend

    @property
    def south_blend(self):
        if self._south_blend is None:
            self._south_blend = img2blender(self.south, self.img_size, self.px_res)
        return self._south_blend

    @property
    def center_blend(self):
        if self._center_blend is None:
            self._center_blend = img2blender(self.center, self.img_size, self.px_res)
        return self._center_blend

    @property
    def rot_matrix(self):
        """
        Rotation matrix from the Blender reference frame to the terrain reference frame
        """
        if self._rot_matrix is None:
            dy = self.north_blend[1] - self.west_blend[1]
            dx = self.north_blend[0] - self.west_blend[0]
            theta = np.arctan2(dy, dx)
            cos = np.cos(theta)
            sin = np.sin(theta)
            self._rot_matrix = np.array(((cos, -sin), (sin, cos)))
        return self._rot_matrix
    
    @property
    def terrain_ortho_size(self):
        # print(f"Inside class: {self._terrain_ortho_size}")
        if self._terrain_ortho_size == [None, None]:
            terrain_ortho_width = np.linalg.norm(self.north_blend - self.west_blend)
            terrain_ortho_height = np.linalg.norm(self.south_blend - self.west_blend)
            if terrain_ortho_width > terrain_ortho_height:
                terrain_ortho_width, terrain_ortho_height = terrain_ortho_height, terrain_ortho_width
            self._terrain_ortho_size[0] = terrain_ortho_width
            self._terrain_ortho_size[1] = terrain_ortho_height

        return self._terrain_ortho_size
          
    def draw_terrain_main_pts(self):
        output_image = self.img.copy()

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(output_image)
        ax.scatter(self.north[0], self.north[1], color='red', s=100, label='North')
        ax.scatter(self.east[0], self.east[1], color='blue', s=100, label='East')
        ax.scatter(self.south[0], self.south[1], color='yellow', s=100, label='South')
        ax.scatter(self.west[0], self.west[1], color='green', s=100, label='West')
        ax.scatter(self.center[0], self.center[1], color='pink', s=100, label='Center')
        ax.legend()
        ax.set_title('Image with Corners')
        plt.show()


    def _find_corner(self, direction):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        img_width, img_height = self.img_size
        if direction == 'north':
            for i in range(img_height):
                if np.any(binary[i, :]):
                    j = np.argmax(binary[i, :])
                    return np.array([j, i])
        elif direction == 'west':
            for j in range(img_width):
                if np.any(binary[:, j]):
                    i = np.argmax(binary[:, j])
                    return np.array([j, i])
        elif direction == 'east':
            for j in range(img_width - 1, -1, -1):
                if np.any(binary[:, j]):
                    i = np.argmax(binary[:, j])
                    return np.array([j, i])
        elif direction == 'south':
            for i in range(img_height - 1, -1, -1):
                if np.any(binary[i, :]):
                    j = np.argmax(binary[i, :])
                    return np.array([j, i])
        return np.array([0, 0])  # Default return in case of failure
   