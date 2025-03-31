import math
import pygame as pg
import numpy as np

from settings import *


def visible_faces(cube_center, cube_size):
    half_size = cube_size / 2

    normals = np.array([
        [0, 0, 1],  # Front
        [0, 0, -1],  # Back
        [-1, 0, 0],  # Right
        [1, 0, 0],  # Left
        [0, 1, 0],  # Top
        [0, -1, 0]  # Bottom
    ])

    face_centers = np.array([
        cube_center + np.array([0, 0, -half_size]),
        cube_center + np.array([0, 0, half_size]),
        cube_center + np.array([half_size, 0, 0]),
        cube_center + np.array([-half_size, 0, 0]),
        cube_center + np.array([0, half_size, 0]),
        cube_center + np.array([0, -half_size, 0]),
    ])

    visible = []

    for face_center, normal in zip(face_centers, normals):
        dot_product = np.dot(face_center, normal)
        visible.append(dot_product > 0)

    return visible


class VoxelRenderer:
    def __init__(self):
        self.surf = pg.display.get_surface()

        self.window_width = self.surf.get_width()
        self.window_height = self.surf.get_height()

    def convert_3d(self, rel_point, abs_point, x_rad, y_rad, z_rad, x_rad_translate, y_rad_translate, z_rad_translate):
        # rel point is the point relative to the object, abs point are the absolute coordinants of the point
        # z translate rotation
        matrix = np.matmul(rel_point, np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([abs_point[0] + matrix[0], abs_point[1] + matrix[1], abs_point[2] + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        return matrix

    def draw_back(self, x, y, z, width, height, depth, x_rad, y_rad, z_rad,
                  x_rad_translate, y_rad_translate, z_rad_translate, color):
        # z translate rotation
        matrix = np.matmul(np.array([-width / 2, -height / 2, depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x1 = matrix[0]
        y1 = matrix[1]
        z1 = matrix[2]

        # z translate rotation
        matrix = np.matmul(np.array([width / 2, -height / 2, depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x2 = matrix[0]
        y2 = matrix[1]
        z2 = matrix[2]

        # z translate rotation
        matrix = np.matmul(np.array([width / 2, height / 2, depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x3 = matrix[0]
        y3 = matrix[1]
        z3 = matrix[2]

        # z translate rotation
        matrix = np.matmul(np.array([-width / 2, height / 2, depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x4 = matrix[0]
        y4 = matrix[1]
        z4 = matrix[2]

        if z1 > 0 and z2 > 0 and z3 > 0 and z4 > 0:
            pg.draw.polygon(
                self.surf,
                color,
                [
                    ((x1 / z1) * FOV + (self.window_width / 2), (y1 / z1) * FOV + (self.window_height / 2)),
                    ((x2 / z2) * FOV + (self.window_width / 2), (y2 / z2) * FOV + (self.window_height / 2)),
                    ((x3 / z3) * FOV + (self.window_width / 2), (y3 / z3) * FOV + (self.window_height / 2)),
                    ((x4 / z4) * FOV + (self.window_width / 2), (y4 / z4) * FOV + (self.window_height / 2))])

    def draw_left(self, x, y, z, width, height, depth, x_rad, y_rad, z_rad,
                  x_rad_translate, y_rad_translate, z_rad_translate, color):
        # z translate rotation
        matrix = np.matmul(np.array([-width / 2, -height / 2, -depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x1 = matrix[0]
        y1 = matrix[1]
        z1 = matrix[2]

        # z translate rotation
        matrix = np.matmul(np.array([-width / 2, -height / 2, depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x2 = matrix[0]
        y2 = matrix[1]
        z2 = matrix[2]

        # z translate rotation
        matrix = np.matmul(np.array([-width / 2, height / 2, depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x3 = matrix[0]
        y3 = matrix[1]
        z3 = matrix[2]

        # z translate rotation
        matrix = np.matmul(np.array([-width / 2, height / 2, -depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x4 = matrix[0]
        y4 = matrix[1]
        z4 = matrix[2]

        if z1 > 0 and z2 > 0 and z3 > 0 and z4 > 0:
            pg.draw.polygon(
                self.surf,
                color,
                [
                    ((x1 / z1) * FOV + (self.window_width / 2), (y1 / z1) * FOV + (self.window_height / 2)),
                    ((x2 / z2) * FOV + (self.window_width / 2), (y2 / z2) * FOV + (self.window_height / 2)),
                    ((x3 / z3) * FOV + (self.window_width / 2), (y3 / z3) * FOV + (self.window_height / 2)),
                    ((x4 / z4) * FOV + (self.window_width / 2), (y4 / z4) * FOV + (self.window_height / 2))])

    def draw_right(self, x, y, z, width, height, depth, x_rad, y_rad, z_rad,
                   x_rad_translate, y_rad_translate, z_rad_translate, color):
        # z translate rotation
        matrix = np.matmul(np.array([width / 2, -height / 2, -depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x1 = matrix[0]
        y1 = matrix[1]
        z1 = matrix[2]

        # z translate rotation
        matrix = np.matmul(np.array([width / 2, -height / 2, depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x2 = matrix[0]
        y2 = matrix[1]
        z2 = matrix[2]

        # z translate rotation
        matrix = np.matmul(np.array([width / 2, height / 2, depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x3 = matrix[0]
        y3 = matrix[1]
        z3 = matrix[2]

        # z translate rotation
        matrix = np.matmul(np.array([width / 2, height / 2, -depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x4 = matrix[0]
        y4 = matrix[1]
        z4 = matrix[2]

        if z1 > 0 and z2 > 0 and z3 > 0 and z4 > 0:
            pg.draw.polygon(
                self.surf,
                color,
                [
                    ((x1 / z1) * FOV + (self.window_width / 2), (y1 / z1) * FOV + (self.window_height / 2)),
                    ((x2 / z2) * FOV + (self.window_width / 2), (y2 / z2) * FOV + (self.window_height / 2)),
                    ((x3 / z3) * FOV + (self.window_width / 2), (y3 / z3) * FOV + (self.window_height / 2)),
                    ((x4 / z4) * FOV + (self.window_width / 2), (y4 / z4) * FOV + (self.window_height / 2))])

    def draw_top(self, x, y, z, width, height, depth, x_rad, y_rad, z_rad,
                 x_rad_translate, y_rad_translate, z_rad_translate, color):
        # z translate rotation
        matrix = np.matmul(np.array([-width / 2, -height / 2, -depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x1 = matrix[0]
        y1 = matrix[1]
        z1 = matrix[2]

        # z translate rotation
        matrix = np.matmul(np.array([width / 2, -height / 2, -depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x2 = matrix[0]
        y2 = matrix[1]
        z2 = matrix[2]

        # z translate rotation
        matrix = np.matmul(np.array([width / 2, -height / 2, depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x3 = matrix[0]
        y3 = matrix[1]
        z3 = matrix[2]

        # z translate rotation
        matrix = np.matmul(np.array([-width / 2, -height / 2, depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x4 = matrix[0]
        y4 = matrix[1]
        z4 = matrix[2]

        if z1 > 0 and z2 > 0 and z3 > 0 and z4 > 0:
            pg.draw.polygon(
                self.surf,
                color,
                [
                    ((x1 / z1) * FOV + (self.window_width / 2), (y1 / z1) * FOV + (self.window_height / 2)),
                    ((x2 / z2) * FOV + (self.window_width / 2), (y2 / z2) * FOV + (self.window_height / 2)),
                    ((x3 / z3) * FOV + (self.window_width / 2), (y3 / z3) * FOV + (self.window_height / 2)),
                    ((x4 / z4) * FOV + (self.window_width / 2), (y4 / z4) * FOV + (self.window_height / 2))])

    def draw_bottom(self, x, y, z, width, height, depth, x_rad, y_rad, z_rad,
                    x_rad_translate, y_rad_translate, z_rad_translate, color):
        # z translate rotation
        matrix = np.matmul(np.array([-width / 2, height / 2, -depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x1 = matrix[0]
        y1 = matrix[1]
        z1 = matrix[2]

        # z translate rotation
        matrix = np.matmul(np.array([width / 2, height / 2, -depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x2 = matrix[0]
        y2 = matrix[1]
        z2 = matrix[2]

        # z translate rotation
        matrix = np.matmul(np.array([width / 2, height / 2, depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x3 = matrix[0]
        y3 = matrix[1]
        z3 = matrix[2]

        # z translate rotation
        matrix = np.matmul(np.array([-width / 2, height / 2, depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0], [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0], [math.sin(z_rad), math.cos(z_rad), 0], [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0], [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad), -math.sin(x_rad)], [0, math.sin(x_rad), math.cos(x_rad)]]))
        x4 = matrix[0]
        y4 = matrix[1]
        z4 = matrix[2]

        if z1 > 0 and z2 > 0 and z3 > 0 and z4 > 0:
            pg.draw.polygon(
                self.surf,
                color,
                [
                    ((x1 / z1) * FOV + (self.window_width / 2), (y1 / z1) * FOV + (self.window_height / 2)),
                    ((x2 / z2) * FOV + (self.window_width / 2), (y2 / z2) * FOV + (self.window_height / 2)),
                    ((x3 / z3) * FOV + (self.window_width / 2), (y3 / z3) * FOV + (self.window_height / 2)),
                    ((x4 / z4) * FOV + (self.window_width / 2), (y4 / z4) * FOV + (self.window_height / 2))])

    def draw_front(self, x, y, z, width, height, depth, x_rad, y_rad, z_rad,
                   x_rad_translate, y_rad_translate, z_rad_translate, color):
        # z translate rotation
        matrix = np.matmul(np.array([-width / 2, -height / 2, -depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0],
             [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)],
             [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0],
             [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0],
             [math.sin(z_rad), math.cos(z_rad), 0],
             [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array([
            [math.cos(y_rad), 0, math.sin(y_rad)],
            [0, 1, 0],
            [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0],
             [0, math.cos(x_rad), -math.sin(x_rad)],
             [0, math.sin(x_rad), math.cos(x_rad)]]))
        x1 = matrix[0]
        y1 = matrix[1]
        z1 = matrix[2]

        # z translate rotation
        matrix = np.matmul(np.array([width / 2, -height / 2, -depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0],
             [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)],
             [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0],
             [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0],
             [math.sin(z_rad), math.cos(z_rad), 0],
             [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array([
            [math.cos(y_rad), 0, math.sin(y_rad)],
            [0, 1, 0],
            [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0],
             [0, math.cos(x_rad), -math.sin(x_rad)],
             [0, math.sin(x_rad), math.cos(x_rad)]]))
        x2 = matrix[0]
        y2 = matrix[1]
        z2 = matrix[2]

        # z translate rotation
        matrix = np.matmul(np.array([width / 2, height / 2, -depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0],
             [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)], [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0], [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0],
             [math.sin(z_rad), math.cos(z_rad), 0],
             [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array([[math.cos(y_rad), 0, math.sin(y_rad)], [0, 1, 0],
                                             [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0],
             [0, math.cos(x_rad), -math.sin(x_rad)],
             [0, math.sin(x_rad), math.cos(x_rad)]]))
        x3 = matrix[0]
        y3 = matrix[1]
        z3 = matrix[2]

        # z translate rotation
        matrix = np.matmul(np.array([-width / 2, height / 2, -depth / 2]), np.array(
            [[math.cos(z_rad_translate), -math.sin(z_rad_translate), 0],
             [math.sin(z_rad_translate), math.cos(z_rad_translate), 0],
             [0, 0, 1]]))
        # y translate rotation
        matrix = np.matmul(matrix, np.array(
            [[math.cos(y_rad_translate), 0, math.sin(y_rad_translate)],
             [0, 1, 0],
             [-math.sin(y_rad_translate), 0, math.cos(y_rad_translate)]]))
        # x translate rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0],
             [0, math.cos(x_rad_translate), -math.sin(x_rad_translate)],
             [0, math.sin(x_rad_translate), math.cos(x_rad_translate)]]))
        # z rotation
        matrix = np.matmul(np.array([x + matrix[0], y + matrix[1], z + matrix[2]]), np.array(
            [[math.cos(z_rad), -math.sin(z_rad), 0],
             [math.sin(z_rad), math.cos(z_rad), 0],
             [0, 0, 1]]))
        # y rotation
        matrix = np.matmul(matrix, np.array([
            [math.cos(y_rad), 0, math.sin(y_rad)],
            [0, 1, 0],
            [-math.sin(y_rad), 0, math.cos(y_rad)]]))
        # x rotation
        matrix = np.matmul(matrix, np.array(
            [[1, 0, 0],
             [0, math.cos(x_rad), -math.sin(x_rad)],
             [0, math.sin(x_rad), math.cos(x_rad)]]))
        x4 = matrix[0]
        y4 = matrix[1]
        z4 = matrix[2]

        if z1 > 0 and z2 > 0 and z3 > 0 and z4 > 0:
            pg.draw.polygon(
                self.surf,
                color,
                [
                    ((x1 / z1) * FOV + (self.window_width / 2), (y1 / z1) * FOV + (self.window_height / 2)),
                    ((x2 / z2) * FOV + (self.window_width / 2), (y2 / z2) * FOV + (self.window_height / 2)),
                    ((x3 / z3) * FOV + (self.window_width / 2), (y3 / z3) * FOV + (self.window_height / 2)),
                    ((x4 / z4) * FOV + (self.window_width / 2), (y4 / z4) * FOV + (self.window_height / 2))])

    def draw_3d_cube(self, x, y, z, width, height, depth, xRot, yRot, zRot,
                     xRotTranslate, yRotTranslate, zRotTranslate, color, unculled_faces):
        if x > FAR * VOXEL_WIDTH or y > FAR * VOXEL_HEIGHT or z > FAR * VOXEL_DEPTH:
            return
        x_rad = math.radians(xRot)
        y_rad = math.radians(yRot)
        z_rad = math.radians(zRot)

        x_rad_translate = math.radians(xRotTranslate)
        y_rad_translate = math.radians(yRotTranslate)
        z_rad_translate = math.radians(zRotTranslate)

        visible_sides = visible_faces(np.array([x, y, z]), width)  # , np.array([cam_pos.x, cam_pos.y, cam_pos.z])

        if visible_sides[1] and unculled_faces['back']:
            self.draw_back(x, y, z, width, height, depth, x_rad, y_rad, z_rad,
                           x_rad_translate, y_rad_translate, z_rad_translate, (max(color[0] - 105, 0),
                                                                                 max(color[1] - 105, 0),
                                                                                 max(color[2] - 105, 0)))

        if visible_sides[5] and unculled_faces['bottom']:
            self.draw_bottom(x, y, z, width, height, depth, x_rad, y_rad, z_rad,
                             x_rad_translate, y_rad_translate, z_rad_translate, (max(color[0] - 155, 0),
                                                                                 max(color[1] - 155, 0),
                                                                                 max(color[2] - 155, 0)))

        if visible_sides[0] and unculled_faces['front']:
            self.draw_front(x, y, z, width, height, depth, x_rad, y_rad, z_rad,
                            x_rad_translate, y_rad_translate, z_rad_translate, (45 if color[0] == 0
                                                                                else color[0], 45 if color[1] == 0 else
                                                                                color[1],
                                                                                45 if color[2] == 0 else color[2]))

        if visible_sides[3] and unculled_faces['left']:
            self.draw_left(x, y, z, width, height, depth, x_rad, y_rad, z_rad,
                           x_rad_translate, y_rad_translate, z_rad_translate, (color[0], color[1], color[2]))

        if visible_sides[2] and unculled_faces['right']:
            self.draw_right(x, y, z, width, height, depth, x_rad, y_rad, z_rad,
                            x_rad_translate, y_rad_translate, z_rad_translate, (max(color[0] - 75, 0),
                                                                                 max(color[1] - 75, 0),
                                                                                 max(color[2] - 75, 0)))

        if visible_sides[4] and unculled_faces['top']:
            self.draw_top(x, y, z, width, height, depth, x_rad, y_rad, z_rad,
                          x_rad_translate, y_rad_translate, z_rad_translate, (90 if color[0] == 0
                                                                                else color[0], 90 if color[1] == 0 else
                                                                                color[1],
                                                                                90 if color[2] == 0 else color[2]))
