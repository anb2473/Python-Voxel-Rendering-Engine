import math
import sys
import pygame as pg
import numpy as np

from pyglm import glm

from camera import Camera
from voxel_renderer import VoxelRenderer
from settings import *


class World:
    def __init__(self, WIN_SIZE=(1200, 700)):
        pg.init()

        pg.display.set_mode(WIN_SIZE)

        pg.mouse.set_visible(False)
        pg.event.set_grab(True)

        self.win_size = WIN_SIZE

        self.surf = pg.display.get_surface()

        self.voxel_renderer = VoxelRenderer()

        self.camera = Camera()

        self.camera.position = glm.vec3(WORLD_WIDTH * CHUNK_WIDTH * VOXEL_WIDTH / 2, 0,
                                        WORLD_DEPTH * CHUNK_DEPTH * VOXEL_DEPTH / 2, )

        self.chunks = []
        for row in range(WORLD_DEPTH):
            self.chunks.append([])
            for column in range(WORLD_WIDTH):
                self.chunks[row].append([])
                for chunk in range(WORLD_HEIGHT):
                    self.chunks[row][column].append([])
                    for z in range(CHUNK_DEPTH):
                        for x in range(CHUNK_WIDTH):
                            self.chunks[row][column][chunk].append([])
                            for y in range(CHUNK_HEIGHT):
                                self.chunks[row][column][chunk][z].append([])
                                self.chunks[row][column][chunk][z][x].append(None)

    def add_voxel(self, x, y, z, voxel_id):
        chunk_row_num = math.floor(z / CHUNK_DEPTH)

        chunk_column = math.floor(x / CHUNK_WIDTH)

        chunk_index = math.floor(y / CHUNK_HEIGHT)

        self.chunks[chunk_row_num][chunk_column][chunk_index][z - chunk_row_num * CHUNK_DEPTH][
            x - chunk_column * CHUNK_WIDTH][y - chunk_index * CHUNK_HEIGHT] = voxel_id

    def add_voxel_array(self):
        pass

    def flatten_list(self, lst):
        flattened = []
        for item in lst:
            if isinstance(item, list) and (len(item) > 0 and type(item[0]) is not int):
                flattened.extend(self.flatten_list(item))
            elif type(item) is list and len(item) > 0:
                flattened.append(item)
        return flattened

    def render_chunk(self, chunk, row_num, column_num, chunk_num):
        flattened_chunk = self.flatten_list(chunk)
        for voxel in sorted(flattened_chunk,
                            key=lambda obj: np.dot(
                                glm.vec3(obj[2] * VOXEL_WIDTH, obj[3] * VOXEL_HEIGHT,
                                         obj[4] * VOXEL_DEPTH) - self.camera.position,
                                -self.camera.forward)):
            self.voxel_renderer.draw_3d_cube(
                voxel[2] * VOXEL_WIDTH + (column_num * CHUNK_WIDTH * VOXEL_WIDTH) - self.camera.position[0],
                voxel[3] * VOXEL_HEIGHT + (chunk_num * CHUNK_HEIGHT * VOXEL_HEIGHT) - self.camera.position[1],
                voxel[4] * VOXEL_DEPTH + (row_num * CHUNK_DEPTH * VOXEL_DEPTH) - self.camera.position[2],
                VOXEL_WIDTH, VOXEL_HEIGHT, VOXEL_DEPTH,
                self.camera.pitch, self.camera.yaw, self.camera.roll,
                0, 0, 0,
                (255, 0, 0), voxel[1])  # , self.camera.position

    def clear(self):
        self.surf.fill('black')

    def cull_face(self):
        row_num = 0
        for row in self.chunks:
            column_num = 0
            for column in row:
                chunk_num = 0
                for chunk in column:
                    z_num = 0
                    for z in chunk:
                        x_num = 0
                        for x in z:
                            y_num = 0
                            for y in x:
                                if y is not None:
                                    # TODO: (a) Test new cull system and (b) check whether the cross chunk exists
                                    visible_faces = {'top': x[y_num - 1] is None, #if y_num - 1 >= 0
                                                     #else column[chunk_num - 1][z_num][x_num][CHUNK_HEIGHT],
                                                     'bottom': x[y_num + 1] is None, #if y_num + 1 < CHUNK_HEIGHT
                                                    # else column[chunk_num + 1][z_num][x_num][0] is None,
                                                     'front': chunk[z_num - 1][x_num][y_num] is None, #if z_num - 1 >= 0
                                                    # else self.chunks[row_num - 1][column_num][chunk_num][CHUNK_DEPTH][
                                                         #x_num][y_num],
                                                     'back': chunk[z_num + 1][x_num][y_num] is None, #if z_num + 1 >= 0
                                                    # else self.chunks[row_num + 1][column_num][chunk_num][0][x_num][
                                                       #  y_num],
                                                     'left': z[x_num - 1][y_num] is None, #if x_num - 1 >= 0 else
                                                  #   row[column_num - 1][chunk_num][z_num][CHUNK_WIDTH][y_num],
                                                     'right': z[x_num + 1][y_num] is None #if x_num + 1 >= 0 else
                                                   #  row[column_num + 1][chunk_num][z_num][0][y_num]
                                                     }
                                    self.chunks[row_num][column_num][chunk_num][z_num][x_num][y_num] = [y,
                                                                                                        visible_faces,
                                                                                                        x_num, y_num,
                                                                                                        z_num]
                                y_num += 1
                            x_num += 1
                        z_num += 1
                    chunk_num += 1
                column_num += 1
            row_num += 1

    def update(self):
        self.camera.update(self.chunks)

    def render(self):
        for event in pg.event.get():
            if event == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                pg.quit()
                sys.exit()
        self.clear()

        row_num = 0
        for row in self.chunks:
            column_num = 0
            for column in row:
                chunk_num = 0
                for chunk in column:
                    self.render_chunk(chunk, row_num, column_num, chunk_num)
                    chunk_num += 1
                column_num += 1
            row_num += 1

        pg.display.flip()
