import math

from world import World

world = World()

for z in range(10):
    for x in range(10):
        for y in range(10):
            if (math.cos(x) + math.cos(z)) * 2 < y:
                world.add_voxel(x + 1, y, z + 1, 1)

world.cull_face()

yRot = 0

while True:
    yRot += 0.5
    world.update()
    world.render()
