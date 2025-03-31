import math
import pygame as pg

from pyglm import glm

from collisions import get_collision
from settings import *


class Camera:
    def __init__(self):
        self.position = glm.vec3(0, 0, 0)
        self.pitch = 0
        self.yaw = 0
        self.roll = 0

        self.forward = glm.vec3(0, 0, 1)
        self.right = glm.vec3(-1, 0, 0)
        self.up = glm.vec3(0, 1, 0)

        self.dir = glm.vec3(0)

        self.collided = None

    def update(self, chunks):
        chunk_row_num = math.floor((self.position.y / VOXEL_HEIGHT) / CHUNK_DEPTH)
        chunk_column = math.floor((self.position.x / VOXEL_WIDTH) / CHUNK_WIDTH)
        chunk_index = math.floor((self.position.z / VOXEL_DEPTH) / CHUNK_HEIGHT)

        chunk = chunks[chunk_row_num][chunk_column][chunk_index]
        self.check_mouse()
        self.minimize_angles()
        self.check_keys()
        self.move(chunk, chunk_row_num, chunk_column, chunk_index)
        self.update_camera_vectors()

    def minimize_angles(self):
        self.yaw = self.yaw % 360
        if self.yaw < 0:
            self.yaw = 360 - abs(self.yaw)

        self.pitch = self.pitch % 360
        if self.pitch < 0:
            self.pitch = 360 - abs(self.yaw)

    def move(self, chunk, chunk_row_num, chunk_column, chunk_index):
        # CHANGES DONE: The yaw calculation has been updated from atan(y / x) to degrees(atan(z / x))
        # TODO: Replace yaw check with a comparison between the collided voxels position and the camera
        if self.dir.z != 0:
            self.move_linearly(self.dir.z)
        #     for z in chunk:
        #         for x in z:
        #             for y in x:
        #                 if y is not None and get_collision(
        #                         y[2] * VOXEL_WIDTH + (chunk_column * CHUNK_WIDTH * VOXEL_WIDTH),
        #                         y[3] * VOXEL_HEIGHT + (chunk_index * CHUNK_HEIGHT * VOXEL_HEIGHT),
        #                         y[4] * VOXEL_DEPTH + (chunk_row_num * CHUNK_DEPTH * VOXEL_DEPTH),
        #                         VOXEL_WIDTH, VOXEL_HEIGHT, VOXEL_DEPTH, self.position.x,
        #                         self.position.y, self.position.z, VOXEL_WIDTH, VOXEL_HEIGHT,
        #                         VOXEL_DEPTH):
        #                     voxel_pos = glm.vec3(y[2] * VOXEL_WIDTH, y[3] * VOXEL_HEIGHT, y[4] * VOXEL_DEPTH)
        #                     diff = self.position - voxel_pos
        #                     try:
        #                         yaw_angle = math.degrees(math.atan(diff.z / diff.x))
        #                         yaw_angle = yaw_angle % 360
        #                         if yaw_angle < 0:
        #                             yaw_angle = 360 - abs(yaw_angle)
        #                         print(yaw_angle, diff)
        #                     except:
        #                         yaw_angle = 0
        #                     # yaw = self.yaw
        #                     # if -self.dir.z < 0:
        #                     #     yaw = 360 - yaw
        #                     if 45 < yaw_angle <= 135:
        #                         self.position.x = y[2] * VOXEL_WIDTH + VOXEL_WIDTH
        #                     elif 135 < yaw_angle <= 225:
        #                         self.position.z = y[4] * VOXEL_DEPTH + VOXEL_DEPTH
        #                     elif 225 < yaw_angle <= 315:
        #                         self.position.x = y[2] * VOXEL_WIDTH + VOXEL_WIDTH
        #                     else:
        #                         self.position.z = y[4] * VOXEL_DEPTH + VOXEL_DEPTH
        #                     # self.move_linearly(-self.dir.z)
        #                     # self.position += glm.normalize(-glm.vec3(math.cos(math.radians(self.yaw)), 0,
        #                     #                                          math.sin(math.radians(self.yaw)))) * SPEED`
        #
        if self.dir.x != 0:
            self.move_horizontally(self.dir.x)
        #     for z in chunk:
        #         for x in z:
        #             for y in x:
        #                 if y is not None and get_collision(
        #                         y[2] * VOXEL_WIDTH + (chunk_column * CHUNK_WIDTH * VOXEL_WIDTH),
        #                         y[3] * VOXEL_HEIGHT + (chunk_index * CHUNK_HEIGHT * VOXEL_HEIGHT),
        #                         y[4] * VOXEL_DEPTH + (chunk_row_num * CHUNK_DEPTH * VOXEL_DEPTH),
        #                         VOXEL_WIDTH, VOXEL_HEIGHT, VOXEL_DEPTH, self.position.x,
        #                         self.position.y, self.position.z, VOXEL_WIDTH, VOXEL_HEIGHT,
        #                         VOXEL_DEPTH):
        #                     pass
        #                     # yaw = self.yaw
        #                     # voxel_pos = glm.vec3(y[2] * VOXEL_WIDTH, y[3] * VOXEL_HEIGHT, y[4] * VOXEL_DEPTH)
        #                     # diff = self.position - voxel_pos
        #                     # try:
        #                     #     yaw_angle = math.degrees(math.atan(diff.z / diff.x))
        #                     # except:
        #                     #     yaw_angle = 0
        #                     # # if -self.dir.x < 0:
        #                     # #     yaw = 360 - yaw
        #                     # if 45 < yaw_angle <= 135:
        #                     #     self.position.z = y[4] * VOXEL_DEPTH + VOXEL_DEPTH
        #                     # elif 135 < yaw_angle <= 225:
        #                     #     self.position.x = y[2] * VOXEL_WIDTH + VOXEL_WIDTH
        #                     # elif 225 < yaw_angle <= 315:
        #                     #     self.position.z = y[4] * VOXEL_DEPTH + VOXEL_DEPTH
        #                     # else:
        #                     #     self.position.x = y[2] * VOXEL_WIDTH + VOXEL_WIDTH
        #                     # self.move_horizontally(-self.dir.x)
        #
        if self.dir.y != 0:
            self.move_vertically(self.dir.y)
        #     # if self.dir.y != 0 and self.dir.x == 0 and self.dir.z == 0:
        #     for z in chunk:
        #         for x in z:
        #             for y in x:
        #                 if y is not None and get_collision(
        #                         y[2] * VOXEL_WIDTH + (chunk_column * CHUNK_WIDTH * VOXEL_WIDTH),
        #                         y[3] * VOXEL_HEIGHT + (chunk_index * CHUNK_HEIGHT * VOXEL_HEIGHT),
        #                         y[4] * VOXEL_DEPTH + (chunk_row_num * CHUNK_DEPTH * VOXEL_DEPTH),
        #                         VOXEL_WIDTH, VOXEL_HEIGHT, VOXEL_DEPTH, self.position.x,
        #                         self.position.y, self.position.z, VOXEL_WIDTH, VOXEL_HEIGHT,
        #                         VOXEL_DEPTH):
        #                     pass
        #                     # voxel_pos = glm.vec3(y[2] * VOXEL_WIDTH, y[3] * VOXEL_HEIGHT, y[4] * VOXEL_DEPTH)
        #                     # diff = self.position - voxel_pos
        #                     # try:
        #                     #     yaw_angle = math.atan(diff.y / diff.z)
        #                     # except:
        #                     #     yaw_angle = 0
        #                     # yaw_angle = math.degrees(yaw_angle)
        #                     # if -self.dir.y > 0:
        #                     #     self.position.y = y[3] * VOXEL_HEIGHT + VOXEL_HEIGHT
        #                     # else:
        #                     #     self.position.y = y[3] * VOXEL_HEIGHT - VOXEL_HEIGHT
        #                     # print(yaw_angle)
        #                     # if 135 < yaw_angle < 225:
        #                     #     # if -self.dir.y > 0:
        #                     #     self.position.y = y[3] * VOXEL_HEIGHT + VOXEL_HEIGHT
        #                     #     # else:
        #                     # elif 315 < yaw_angle < 45:
        #                     #     self.position.y = y[3] * VOXEL_HEIGHT - VOXEL_HEIGHT

    def check_keys(self):
        keys = pg.key.get_pressed()
        if keys[pg.K_w]:
            if not keys[pg.K_s]:
                self.dir.z = 1
        elif keys[pg.K_s]:
            self.dir.z = -1
        else:
            self.dir.z = 0

        if keys[pg.K_a]:
            if not keys[pg.K_d]:
                self.dir.x = 1
        elif keys[pg.K_d]:
            self.dir.x = -1
        else:
            self.dir.x = 0

        if keys[pg.K_q]:
            if not keys[pg.K_e]:
                self.dir.y = -1
        elif keys[pg.K_e]:
            self.dir.y = 1
        else:
            self.dir.y = 0

    def move_vertically(self, mult):
        self.position += self.up * mult * SPEED

    def move_horizontally(self, mult):
        self.position += self.right * mult * SPEED

    def move_linearly(self, mult):
        self.position += self.forward * mult * SPEED

    def check_mouse(self):
        mx, my = pg.mouse.get_rel()
        self.yaw += mx * SENSITIVITY
        self.pitch -= my * SENSITIVITY

    def update_camera_vectors(self):
        yaw = glm.radians(self.yaw)

        self.forward.z = glm.cos(yaw)
        self.forward.x = glm.sin(yaw)

        self.forward = glm.normalize(self.forward)
        self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0, 1, 0)))
        self.up = glm.normalize(glm.cross(self.right, self.forward))
