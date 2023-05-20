import math

import numpy as np
from numpy import array
from numpy import sin
from numpy import cos

import MyMath


class RawModel:
    def __init__(self, vertices, colors):
        self.vertices = vertices
        self.colors = colors

    def rotate(self, rotation):
        new_vertices = np.append(self.vertices, np.ones((len(self.vertices), 1)), axis=1)
        new_vertices = new_vertices @ MyMath.createRotationMatrix(rotation)
        new_vertices = np.delete(new_vertices, 3, axis=1)
        self.vertices = new_vertices

    def translate(self, translation):
        self.vertices = self.vertices+translation

    def add(self, other):
        self.vertices = np.vstack([self.vertices, other.vertices])
        self.colors = np.vstack([self.colors, other.colors])


def generateCylinder(detail_circle, ring_list, radius_list, color_list):
    vertices = np.zeros(shape=(0, 3))
    colors = np.zeros(shape=(0, 3))
    for layer in range(len(ring_list) - 1):
        for circle in range(detail_circle):
            x1 = sin(2 * math.pi * (circle / detail_circle)) * radius_list[layer][0]
            z1 = cos(2 * math.pi * (circle / detail_circle)) * radius_list[layer][1]
            y1 = ring_list[layer]
            p1 = array([x1, y1, z1])

            x2 = sin(2 * math.pi * ((circle + 1) / detail_circle)) * radius_list[layer][0]
            z2 = cos(2 * math.pi * ((circle + 1) / detail_circle)) * radius_list[layer][1]
            y2 = ring_list[layer]
            p2 = array([x2, y2, z2])

            x3 = sin(2 * math.pi * ((circle + 1) / detail_circle)) * radius_list[layer + 1][0]
            z3 = cos(2 * math.pi * ((circle + 1) / detail_circle)) * radius_list[layer + 1][1]
            y3 = ring_list[layer + 1]
            p3 = array([x3, y3, z3])

            x4 = sin(2 * math.pi * (circle / detail_circle)) * radius_list[layer + 1][0]
            z4 = cos(2 * math.pi * (circle / detail_circle)) * radius_list[layer + 1][1]
            y4 = ring_list[layer + 1]
            p4 = array([x4, y4, z4])

            vertices = np.vstack([vertices, p4])
            vertices = np.vstack([vertices, p2])
            vertices = np.vstack([vertices, p1])
            colors = np.vstack([colors, color_list])

            vertices = np.vstack([vertices, p4])
            vertices = np.vstack([vertices, p3])
            vertices = np.vstack([vertices, p2])
            colors = np.vstack([colors, color_list])

    return RawModel(vertices, colors)


def generateBox(scale_bot, scale_top, offset_top, color):
    vertices = np.zeros(shape=(0, 3))
    colors = np.zeros(shape=(0, 3))

    p = [[-scale_bot[0], -scale_bot[1], scale_bot[2]], [scale_bot[0], -scale_bot[1], scale_bot[2]],
         [scale_bot[0], -scale_bot[1], -scale_bot[2]], [-scale_bot[0], -scale_bot[1], -scale_bot[2]],
         [-scale_top[0] + offset_top[0], scale_top[1] + offset_top[1], scale_top[2] + offset_top[2]],
         [scale_top[0] + offset_top[0], scale_top[1] + offset_top[1], scale_top[2] + offset_top[2]],
         [scale_top[0] + offset_top[0], scale_top[1] + offset_top[1], -scale_top[2] + offset_top[2]],
         [-scale_top[0] + offset_top[0], scale_top[1] + offset_top[1], -scale_top[2] + offset_top[2]]]

    ##front
    vertices = np.vstack([vertices, array([p[0], p[4], p[5]])])
    colors = np.vstack([colors, color])
    vertices = np.vstack([vertices, array([p[0], p[5], p[1]])])
    colors = np.vstack([colors, color])

    ##back
    vertices = np.vstack([vertices, array([p[3], p[2], p[7]])])
    colors = np.vstack([colors, color])
    vertices = np.vstack([vertices, array([p[6], p[7], p[2]])])
    colors = np.vstack([colors, color])

    ##right
    vertices = np.vstack([vertices, array([p[1], p[5], p[2]])])
    colors = np.vstack([colors, color])
    vertices = np.vstack([vertices, array([p[6], p[2], p[5]])])
    colors = np.vstack([colors, color])

    ##left
    vertices = np.vstack([vertices, array([p[0], p[3], p[4]])])
    colors = np.vstack([colors, color])
    vertices = np.vstack([vertices, array([p[7], p[4], p[3]])])
    colors = np.vstack([colors, color])

    ##top
    vertices = np.vstack([vertices, array([p[7], p[6], p[5]])])
    colors = np.vstack([colors, color])
    vertices = np.vstack([vertices, array([p[7], p[5], p[4]])])
    colors = np.vstack([colors, color])

    return RawModel(vertices, colors)
