import math

import numpy as np
import pygame.math
from numpy import array
from numpy import sin
from numpy import cos
import Noise
import random
from pygame.math import clamp
import MyMath
from math import pi


class Model:
    def __init__(self, position, vertices, colors):
        self.position = position
        self.scale = array([1.0, 1.0, 1.0])
        self.vertices = vertices
        self.rotation_matrix = MyMath.createRotationMatrix([0.0, 0.0, 0.0])
        self.transformed_vertices = np.append(vertices, np.zeros((len(vertices), 1)), axis=1)
        self.transformed_vertices = self.transformed_vertices @ self.rotation_matrix
        self.colors = colors
        self.last_calculated_colors = np.ones(shape=(len(colors), 3))
        self.normals = np.ones((int(len(vertices) / 3), 3))
        for i in range(len(self.normals)):
            v1 = vertices[i * 3 + 1] - vertices[i * 3 + 0]
            v2 = vertices[i * 3 + 2] - vertices[i * 3 + 0]
            normal = np.cross(v1, v2)
            self.normals[i] = normal / np.linalg.norm(normal)

    def rotate(self, rotation):
        rotation = np.append(rotation, 1.0) @ self.rotation_matrix
        self.rotation_matrix = self.rotation_matrix @ MyMath.createRotationMatrix(rotation)


class RawModel:
    def __init__(self, vertices, colors):
        self.vertices = vertices
        self.colors = colors

    def scale(self, scale):
        new_vertices = np.append(self.vertices, np.ones((len(self.vertices), 1)), axis=1)
        new_vertices = new_vertices @ MyMath.createScaleMatrix(scale)
        new_vertices = np.delete(new_vertices, 3, axis=1)
        self.vertices = new_vertices

    def rotate(self, rotation):
        new_vertices = np.append(self.vertices, np.ones((len(self.vertices), 1)), axis=1)
        new_vertices = new_vertices @ MyMath.createRotationMatrix(rotation)
        new_vertices = np.delete(new_vertices, 3, axis=1)
        self.vertices = new_vertices

    def translate(self, translation):
        self.vertices = self.vertices + translation

    def add(self, other):
        self.vertices = np.vstack([self.vertices, other.vertices])
        self.colors = np.vstack([self.colors, other.colors])


def generateTerrainChunkModel(start_x, start_y, size):
    heights = [[0 for x in range(size)] for y in range(size)]
    for y in range(size):
        for x in range(size):
            heights[x][y] = Noise.noiseFunction(start_x + x, start_y + y)



    vertices = np.zeros(shape=(0, 3))
    colors = np.zeros(shape=(0, 3))
    for y in range(size - 1):
        for x in range(size - 1):

            v0 = np.array([x, heights[x][y], y])
            v1 = np.array([x + 1, heights[x + 1][y], y])
            v2 = np.array([x + 1, heights[x + 1][y + 1], y + 1])
            v3 = np.array([x, heights[x][y + 1], y + 1])


            polygon_1 = array([v1, v2, v3])
            polygon_2 = array([v0, v1, v3])

            water_color = array([33, 98, 227])
            grass_color = array([72, 144, 48])

            stem_colour = array([115, 50, 20])
            soil_color = array([85, 47, 18])
            top_colour = array([124, 137, 15])

            ground_color = MyMath.lerp(Noise.noise(0.1*(start_x + x), 0.1*(start_y + y))*1, grass_color, soil_color)
            snow_color = array([236, 255, 253])

            center_height = (v1[1] + v2[1] + v3[1]) / 3
            snow_color = MyMath.lerp(pygame.math.clamp(1-(center_height-1.2), 0 , 1), snow_color, ground_color)

            color = ground_color
            if np.amax(polygon_1[:, 1]) <= 0.0:
                color = water_color * ((center_height / 20) + 0.5)

            if center_height > 1.2:
                color = snow_color
            colors = np.vstack([colors, color])

            center_height = (v0[1] + v1[1] + v3[1]) / 3
            color = ground_color
            if np.amax(polygon_2[:, 1]) <= 0.0:
                color = water_color * ((center_height / 20) + 0.5)

            if center_height > 1.2:
                color = snow_color
            colors = np.vstack([colors, color])

            vertices = np.vstack([vertices, polygon_1])
            vertices = np.vstack([vertices, polygon_2])

    vertices[:, 1] = np.clip(vertices[:, 1], 0, 99999)  # plattar till vattnet

    stem_h = 0.5
    stem_r = 0.03
    p = [[-stem_r, stem_h, stem_r], [stem_r, stem_h, stem_r], [0, stem_h, -stem_r],
         [-stem_r, 0, stem_r], [stem_r, 0, stem_r], [0, 0, -stem_r]]

    tree_vertices = np.zeros(shape=(0, 3))
    tree_colors = np.zeros(shape=(0, 3))

    tree_vertices = np.vstack([tree_vertices, array([p[0], p[3], p[4]])])
    tree_colors = np.vstack([tree_colors, stem_colour])
    tree_vertices = np.vstack([tree_vertices, array([p[0], p[4], p[1]])])
    tree_colors = np.vstack([tree_colors, stem_colour])

    tree_vertices = np.vstack([tree_vertices, array([p[1], p[4], p[5]])])
    tree_colors = np.vstack([tree_colors, stem_colour])
    tree_vertices = np.vstack([tree_vertices, array([p[1], p[5], p[2]])])
    tree_colors = np.vstack([tree_colors, stem_colour])

    tree_vertices = np.vstack([tree_vertices, array([p[0], p[5], p[3]])])
    tree_colors = np.vstack([tree_colors, stem_colour])
    tree_vertices = np.vstack([tree_vertices, array([p[0], p[2], p[5]])])
    tree_colors = np.vstack([tree_colors, stem_colour])

    top_h = 0.4
    top_r = 0.1
    p2 = [[0, top_h + stem_h, 0],
          [-top_r, stem_h, top_r], [top_r, stem_h, top_r], [0, stem_h, -top_r]]

    tree_vertices = np.vstack([tree_vertices, array([p2[0], p2[2], p2[1]])])
    tree_colors = np.vstack([tree_colors, top_colour])

    tree_vertices = np.vstack([tree_vertices, array([p2[0], p2[3], p2[2]])])
    tree_colors = np.vstack([tree_colors, top_colour])

    tree_vertices = np.vstack([tree_vertices, array([p2[0], p2[1], p2[3]])])
    tree_colors = np.vstack([tree_colors, top_colour])

    random.seed(start_x * start_y)
    for y in range(size - 1):
        for x in range(size - 1):

            if 0.1 <= heights[x][y] <= 1.0:
                x_world = x + random.uniform(0, 1)
                y_world = y + random.uniform(0, 1)
                value = Noise.noise((x) * 0.02137, (y) * 0.02137)
                value = pow(abs(value), 1.4)
                if value < 1.2 - random.uniform(0, 0.95):
                    continue
                v0 = np.array([x, heights[x][y], y])
                v1 = np.array([x + 1, heights[x + 1][y], y])
                v2 = np.array([x + 1, heights[x + 1][y + 1], y + 1])
                v3 = np.array([x, heights[x][y + 1], y + 1])
                center_height = MyMath.barryCentric(v0, v1, v3, array([x_world, y_world]))
                r_scale = random.uniform(0, 1.2)
                h_scale = random.uniform(0, 1.2)
                y_rot = random.uniform(0, pi * 2)
                variance = 0.45
                color_variance = array([1 + random.uniform(-variance, variance),
                                        1 + random.uniform(-variance, variance),
                                        1 + random.uniform(-variance, variance)])
                tree = np.append(tree_vertices, np.ones((len(tree_vertices), 1)), axis=1) @ MyMath.createScaleMatrix(
                    array([1 + r_scale, 1 + h_scale, 1 + r_scale]))
                tree = tree @ MyMath.createRotationMatrix(array([0, y_rot, 0]))
                vertices = np.vstack([vertices, np.delete(tree, 3, axis=1) + array([x_world, center_height, y_world])])
                colors = np.vstack([colors, tree_colors * color_variance])

    terrain_model = Model(array([start_x, 0.0, start_y]), vertices, colors)
    return terrain_model


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


def generateAirplane():
    body_color = array([215, 42, 24])
    wing_color = array([199, 209, 222])
    window_color = array([165, 241, 255])

    h_ration = 1.15
    ring_list = [0.0, 0.2, 0.5, 3.5, 4.0]
    radius_list = [[0.01, 0.01 * h_ration], [0.2, 0.2 * h_ration], [0.25, 0.4 * h_ration],
                   [0.25, 0.25 * h_ration], [0.1, 0.1 * h_ration]]

    raw_model = generateCylinder(9, ring_list, radius_list, body_color)
    raw_model.rotate([0, pi, 0])
    raw_model.rotate([pi / 2, 0, 0])
    raw_model.colors[(9 * 2) * 1 + 9] = window_color
    raw_model.colors[(9 * 2) * 1 + 9 - 1] = window_color
    raw_model.colors[(9 * 2) * 1 + 9 - 2] = window_color
    raw_model.colors[(9 * 2) * 1 + 9 - 3] = window_color
    raw_model.colors[(9 * 2) * 1 + 9 + 1] = window_color
    raw_model.colors[(9 * 2) * 1 + 9 + 2] = window_color

    for i in range(len(raw_model.vertices)):
        if raw_model.vertices[i][1] < -0.20:
            raw_model.colors[int(i / 3)] = wing_color

    wing1 = generateBox([0.02, 0, 0.45], [0.02, 1.7, 0.15], [0, 0, -0.4], wing_color)
    wing1.rotate([0, 0, pi / 2])
    wing1.translate(array([0.25, 0, -2]))
    raw_model.add(wing1)

    wing2 = generateBox([0.02, 0, 0.45], [0.02, 1.7, 0.15], [0, 0, -0.4], wing_color)
    wing2.rotate([0, 0, -pi / 2])
    wing2.translate(array([-0.25, 0, -2]))
    raw_model.add(wing2)

    wing1 = generateBox([0.02, 0, 0.25], [0.02, .4, 0.08], [0, 0.0, -0.25], wing_color)
    wing1.rotate([0, 0, pi / 2])
    wing1.translate(array([0.15, 0.25, -3.6]))
    raw_model.add(wing1)

    wing1 = generateBox([0.02, 0, 0.25], [0.02, .4, 0.08], [0, 0.0, -0.25], wing_color)
    wing1.rotate([0, 0, -pi / 2])
    wing1.translate(array([-0.15, 0.25, -3.6]))
    raw_model.add(wing1)

    wing1 = generateBox([0.02, 0, 0.25], [0.02, .4, 0.08], [0, 0.0, -0.25], body_color)
    wing1.translate(array([0, 0.25, -3.6]))
    raw_model.add(wing1)

    wing1 = generateBox([0.02, 0, 0.15], [0.02, .4, 0.04], [0, 0.0, -0.25], body_color)
    wing1.translate(array([1.93, 0.02, -2.4]))
    raw_model.add(wing1)

    wing1 = generateBox([0.02, 0, 0.15], [0.02, .4, 0.04], [0, 0.0, -0.25], body_color)
    wing1.translate(array([-1.93, 0.02, -2.4]))
    raw_model.add(wing1)


    ring_list = [0.0, 0.05, 0.35, 0.6]
    radius_list = [[0.05, 0.05], [0.20, 0.20 ], [0.15, 0.15], [0.10, 0.10]]
    engine1 = generateCylinder(6, ring_list, radius_list, wing_color)
    engine1.rotate([math.pi/2, 0, 0])
    engine1.translate(array([-0.73, -0.30, -1.95]))
    for i in range(len(engine1.vertices)):
        if engine1.vertices[i][2] < -2.5:
            engine1.colors[int(i / 3)] = body_color
    raw_model.add(engine1)

    engine2 = generateCylinder(6, ring_list, radius_list, wing_color)
    engine2.rotate([math.pi/2, 0, 0])
    engine2.translate(array([0.73, -0.30, -1.95]))
    for i in range(len(engine2.vertices)):
        if engine2.vertices[i][2] < -2.5:
            engine2.colors[int(i / 3)] = body_color
    raw_model.add(engine2)

    raw_model.scale(array([0.5, 0.5, 0.5]))

    plane = Model(array([0, 0, 0]), raw_model.vertices, raw_model.colors)
    return plane
