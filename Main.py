import math
import numpy as np
import pygame
import time
import random
from math import pi
import numpy
from pygame.math import clamp
from numpy import array
from numpy import cos
from numpy import sin
import Noise
from pygame.locals import *  ## allt för att hämta konstanter till varje tangent

run = True
width = 1800
height = 950
display = pygame.display.set_mode((width, height))
far_plane = 60
near_plane = 0.1
delta = 0
last_frame_time = time.time()
last_second_time = time.time()
frames_current_second = 0
frames_last_second = 0


class Chunk:
    chunk_size = 3

    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
        self.center_world_pos = array(
            [self.x * self.chunk_size + self.chunk_size / 2, self.y * self.chunk_size + self.chunk_size / 2])
        model = generateTerrainChunkModel(self.x * self.chunk_size, self.y * self.chunk_size, self.chunk_size + 1)
        self.model = model
        models.append(model)

    def cleanUp(self):
        models.remove(self.model)


class Model:
    def __init__(self, position, vertices, colors):
        self.position = position
        self.scale = array([1.0, 1.0, 1.0])
        self.vertices = vertices
        self.colors = colors
        self.rotation = array([0.0, 0.0, 0.0])
        self.normals = np.zeros((int(len(vertices) / 3), 3))
        for i in range(len(self.normals)):
            v1 = vertices[i * 3 + 1] - vertices[i * 3 + 0]
            v2 = vertices[i * 3 + 2] - vertices[i * 3 + 0]
            normal = np.cross(v1, v2)
            self.normals[i] = normal / np.linalg.norm(normal)


class Polygon_2D:
    def __init__(self, vertices, color, depth):
        self.vertices = vertices
        self.color = color
        self.depth = depth


class Camera:
    def __init__(self):
        self.position = array([0.0, 0.0, 0.0])
        self.rotation = array([0.0, 0, 0.0, 0.0])


camera = Camera()
camera.position[0] = 250
camera.position[1] = 3
camera.position[0] = 250
models = []
chunks = []


def calcDelta():
    global last_frame_time
    global last_second_time
    global frames_current_second
    global delta
    current_frame_time = time.time()
    delta = current_frame_time - last_frame_time
    if time.time() - last_second_time >= 1:
        global frames_last_second
        frames_last_second = frames_current_second
        last_second_time = current_frame_time
        frames_current_second = 0
        pygame.display.set_caption("Flight-Simulator-3D " + frames_last_second.__str__() + "fps")
    frames_current_second += 1
    last_frame_time = current_frame_time
    delta *= 5


def handleInput():
    speed = 1.5
    turnspeed = 0.6
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            global run
            run = False
        if event.type == MOUSEMOTION:
            movement = pygame.mouse.get_rel()
            camera.rotation[2] += movement[0] * delta * 0.01
            camera.rotation[0] -= movement[1] * delta * 0.01
    keys = pygame.key.get_pressed()

    if keys[K_w]:
        camera.position[0] -= speed * delta * sin(-camera.rotation[1])
        camera.position[2] -= speed * delta * cos(-camera.rotation[1])
        camera.position[1] += speed * delta * sin(-camera.rotation[0])
    if keys[K_s]:
        camera.position[0] += speed * delta * sin(-camera.rotation[1])
        camera.position[2] += speed * delta * cos(-camera.rotation[1])
        camera.position[1] -= speed * delta * sin(-camera.rotation[0])
    if keys[K_a]:
        camera.rotation[1] -= turnspeed * delta
    if keys[K_d]:
        camera.rotation[1] += turnspeed * delta
    if keys[K_q]:
        camera.position[1] -= speed * delta
    if keys[K_e]:
        camera.position[1] += speed * delta
    if keys[K_r]:
        camera.rotation[0] -= turnspeed * delta
    if keys[K_f]:
        camera.rotation[0] += turnspeed * delta
    if keys[K_c]:
        camera.rotation[2] += turnspeed * delta
    if keys[K_v]:
        camera.rotation[2] -= turnspeed * delta


def createRotationMatrix(rotation):
    rotation_x_matrix = array([[1, 0, 0, 0],
                               [0, cos(rotation[0]), -sin(rotation[0]), 0],
                               [0, sin(rotation[0]), cos(rotation[0]), 0],
                               [0, 0, 0, 1]])
    rotation_y_matrix = array([[cos(rotation[1]), 0, sin(rotation[1]), 0],
                               [0, 1, 0, 0],
                               [-sin(rotation[1]), 0, cos(rotation[1]), 0],
                               [0, 0, 0, 0]])
    rotation_z_matrix = array([[cos(rotation[2]), -sin(rotation[2]), 0, 0],
                               [sin(rotation[2]), cos(rotation[2]), 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 0]])
    return rotation_x_matrix @ rotation_y_matrix @ rotation_z_matrix


def createTranslateMatrix(translation):
    translate_matrix = numpy.identity(4)
    translate_matrix[0:3, 3] = translation
    return translate_matrix


def createViewMatrix(camera):
    matrix = numpy.identity(4)
    matrix = createTranslateMatrix(-camera.position) @ matrix
    matrix = createRotationMatrix(camera.rotation) @ matrix
    return matrix


def createProjectionMatrix():
    field_of_view = pi / 2  # 90 grader
    aspect_ratio = display.get_width() / display.get_height()
    y_scale = 1 / numpy.arctan(field_of_view / 2)
    x_scale = y_scale / aspect_ratio
    frustum_length = far_plane - near_plane
    A = -((far_plane + near_plane) / frustum_length)
    B = -((2 * far_plane * near_plane) / frustum_length)
    projection_matrix = array([[x_scale, 0, 0, 0],
                               [0, y_scale, 0, 0],
                               [0, 0, A, B],
                               [0, 0, -1, 0]])
    return projection_matrix


view_matrix = createViewMatrix(camera)
projection_matrix = createProjectionMatrix()


def fromWorldToScreen(point_3D):
    clipspace_position = projection_matrix @ view_matrix @ point_3D
    if clipspace_position[3] == 0: clipspace_position[3] = 0.00001
    NDC_space = clipspace_position[0:2] / clipspace_position[3]
    x = ((NDC_space[0] + 1) / 2) * display.get_width()
    y = ((NDC_space[1] + 1) / 2) * display.get_height()
    screen_space = array([x, y, clipspace_position[2]])
    return screen_space


def getDistanceBetween_2D(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


def hasChunk(x, y):
    for chunk in chunks:
        if chunk.x == x and chunk.y == y: return True
    return False


def update():
    global view_matrix
    view_matrix = createViewMatrix(camera)
    calcDelta()

    load_chunk_world_distance = 8
    distance = load_chunk_world_distance
    x = camera.position[0] - distance * sin(-camera.rotation[1])
    z = camera.position[2] - distance * cos(-camera.rotation[1])
    target_point = array([x / Chunk.chunk_size, z / Chunk.chunk_size])
    for i in range(len(chunks) - 1, - 1, -1):
        if getDistanceBetween_2D(chunks[i].center_world_pos,
                                 array([x, z])) > load_chunk_world_distance + Chunk.chunk_size * 2:
            chunks[i].cleanUp()
            chunks.remove(chunks[i])

    r = int(load_chunk_world_distance / Chunk.chunk_size)
    target_x = int(target_point[0])
    target_y = int(target_point[1])
    i = 0
    for y in range(-r, r, 1):
        for x in range(-r, r, 1):
            if not hasChunk(target_x + x, target_y + y):
                chunks.append(Chunk(target_x + x, target_y + y))
                i += 1
    if i > 0:
        print(f"added {i} chunks")


def render():
    polygons = []
    sky_color = array([178, 255, 255])
    display.fill(sky_color)
    light_direction = array([0.9, -0.7, .9])
    light_direction = light_direction / np.linalg.norm(light_direction)
    for model in models:
        scale_matrix = np.identity(4)
        scale_matrix[0, 0] = model.scale[0]
        scale_matrix[1, 1] = model.scale[1]
        scale_matrix[2, 2] = model.scale[2]

        transformed_vertices = numpy.append(model.vertices, np.ones((len(model.vertices), 1)), axis=1)

        transformed_vertices = transformed_vertices @ createRotationMatrix(model.rotation)
        transformed_vertices = transformed_vertices @ scale_matrix
        transformed_vertices = transformed_vertices + (np.append(model.position, 1.0) @ scale_matrix)

        transformed_normals = numpy.append(model.normals, np.ones((len(model.normals), 1)), axis=1)
        transformed_normals = transformed_normals @ createRotationMatrix(model.rotation)
        for i in range(int(len(transformed_vertices) / 3)):
            v1 = fromWorldToScreen(transformed_vertices[i * 3 + 0])
            v2 = fromWorldToScreen(transformed_vertices[i * 3 + 1])
            v3 = fromWorldToScreen(transformed_vertices[i * 3 + 2])
            depth = -(v1[2] + v2[2] + v3[2]) / 3
            if v1[2] < near_plane or v2[2] < near_plane or v3[2] < near_plane:
                continue
            else:
                vertices = array([v1[0:2], v2[0:2], v3[0:2]])

                lighting = np.dot(transformed_normals[i][0:3], light_direction)
                lighting = max(lighting, 0.0) + 0.4
                fog = 1 - array([0.1, 0.1, 0.1]) * v3[2] * 0.1
                colors = model.colors[i] * fog * lighting
                colors = np.clip(colors, 0, 255)

                polygons.append(Polygon_2D(vertices, colors, depth))

    polygons.sort(key=lambda x: x.depth)

    for polygon in polygons:
        polygon.vertices[0][1] = height - polygon.vertices[0][1]
        polygon.vertices[1][1] = height - polygon.vertices[1][1]
        polygon.vertices[2][1] = height - polygon.vertices[2][1]
        pygame.draw.polygon(display, polygon.color, polygon.vertices)
    #drawMap(array([int(width / 2), int(height / 2)]), 200, array([camera.position[0], camera.position[2]]))
    pygame.display.update()
    pygame.display.flip()


def noiseFunction(x, y):
    iterations = 1
    frequency = 1.0
    amplitude = Noise.interpolated_noise(x * 0.1, y * 0.1) * 25
    value = 0
    for i in range(iterations):
        amount = Noise.perlin_noise(x * frequency, y * frequency, 0.5, 2) * amplitude
        amount = 1.1 * (1.5 - abs(0.75 - amount))
        amount = pow(abs(amount), 1.35)

        value += amount
        amplitude = amplitude * 0.25
        frequency = frequency * 3
    value -= Noise.perlin_noise(x * 0.1, y * 0.1, 0.5, 1) * 5

    return value


def generateTerrainChunkModel(start_x, start_y, size):
    heights = [[0 for x in range(size)] for y in range(size)]
    for y in range(size):
        for x in range(size):
            heights[x][y] = noiseFunction(start_x + x, start_y + y)
    vertices = np.zeros(shape=(0, 3))
    colors = np.zeros(shape=(0, 3))
    for y in range(size - 1):
        for x in range(size - 1):

            v0 = np.array([x, heights[x][y], y])
            v1 = np.array([x + 1, heights[x + 1][y], y])
            v2 = np.array([x + 1, heights[x + 1][y + 1], y + 1])
            v3 = np.array([x, heights[x][y + 1], y + 1])

            center_height = (v1[1] + v2[1] + v3[1]) / 3

            snow_color = array([236, 255, 253])
            water_color = array([33, 98, 227]) * (-clamp(center_height, -1, -0.5))
            grass_color = array([72, 144, 48])

            polygon_1= array([v1, v2, v3])
            polygon_2 = array([v0, v1, v3])

            color = grass_color
            if np.amax(polygon_1[:,1]) <= 0.0:
                color = water_color

            if center_height > 1.2:
                color = snow_color
            colors = np.vstack([colors, color])

            center_height = (v0[1] + v1[1] + v3[1]) / 3
            color = grass_color
            if np.amax(polygon_2[:,1]) <= 0.0:
                color = water_color

            if center_height > 1.2:
                color = snow_color
            colors = np.vstack([colors, color])

            vertices = np.vstack([vertices, polygon_1])
            vertices = np.vstack([vertices, polygon_2])

    print(vertices)
    vertices[:, 1] = np.clip(vertices[:, 1], 0, 99999) # plattar till vattnet

    terrain_model = Model(array([start_x, 0.0, start_y]), vertices, colors)
    return terrain_model


def drawMap(position_screen, resolution, position_world):
    r = int(resolution / 2)
    zoom_out = 1
    for x in range(-r, r, 1):
        for y in range(-r, r, 1):
            color = noiseFunction(x * zoom_out + position_world[0], y * zoom_out + position_world[1])
            color = clamp(((color + 1.0) / 2) * 100, 0, 255)
            display.set_at((x + position_screen[0], y + position_screen[1]), [color, 0, 0])


def program():
    for i in range(0):
        position = array(
            [random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0), random.uniform(-10.0, 10.0)])
        r = random.uniform(1.0, 3.0)
        p = array([[-r, r, r], [r, r, r], [r, r, -r], [-r, r, -r],
                   [-r, -r, r], [r, -r, r], [r, -r, -r], [-r, -r, -r]])

        vertices = array([p[0], p[4], p[5],
                          p[0], p[5], p[1],
                          p[1], p[5], p[6],
                          p[1], p[6], p[2],
                          p[3], p[2], p[6],
                          p[3], p[6], p[7],
                          p[0], p[7], p[4],
                          p[0], p[3], p[7],
                          p[3], p[0], p[1],
                          p[3], p[1], p[2],
                          p[7], p[5], p[4],
                          p[7], p[6], p[5]])

        color1 = array([255.0, 0.0, 0.0])
        color2 = array([0.0, 255.0, 0.0])
        color3 = array([0.0, 0.0, 255.0])

        colors = array([color1,
                        color1,
                        color3,
                        color3,
                        color2,
                        color2,
                        color1,
                        color1,
                        color3,
                        color3,
                        color2,
                        color2])

        models.append(Model(position, vertices, colors))

    while run:
        handleInput()
        update()
        render()
    pygame.quit()


program()
