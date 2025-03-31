def get_collision(x1, y1, z1, width1, height1, depth1, x2, y2, z2, width2, height2, depth2):
    return (x1 <= x2 + width2 and x1 + width1 >= x2 and
            y1 <= y2 + height2 and y1 + height1 >= y2 and
            z1 <= z2 + depth2 and z1 + depth1 >= z2)


def get_point_collision(x1, y1, z1, x2, y2, z2, width, height, depth):
    return (x2 <= x1 <= x2 + width and
            y2 <= y1 <= y2 + height and
            z2 <= z1 <= z2 + depth)
