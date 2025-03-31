import math
import random

from world import World

world = World()

for x in range(10):
    for y in range(26):
        world.add_voxel(x + 1, y + 1, 10, y)

world.cull_face()

yRot = 0

while True:
    yRot += 0.5
    world.update()
    world.render()
