import pygame
import time
import random
from math import pi
import numpy
from pygame.math import clamp
from numpy import array
from numpy import cos
from numpy import sin
from pygame.locals import *  ## allt för att hämta konstanter till varje tangent

run = True
width = 800
height = 800
display = pygame.display.set_mode((width, height))
far_plane = 100
near_plane = 1
delta = 0
last_frame_time = time.time()
last_second_time = time.time()
frames_current_second = 0
frames_last_second = 0

class Model:
    def __init__(self, position, vertices, colors, rotation=array([0.0, 0.0, 0.0]),
                 scale=array([1.0, 1.0, 1.0])):
        self.position = position
        self.scale = scale
        self.vertices = vertices
        self.colors = colors
        self.rotation = rotation


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
camera.position[2] = 10
models = []

def calcDelta():
    global last_frame_time
    global last_second_time
    global frames_current_second
    global delta
    current_frame_time = time.time()
    delta = current_frame_time - last_frame_time
    if time.time() - last_second_time >= 1:
        frames_last_second = frames_current_second
        last_second_time = current_frame_time
        frames_current_second = 0
        # print("FPS: " + frames_last_second.__str__())
        # print("Delta: " + delta.__str__())
        pygame.display.set_caption('3D test ' + frames_last_second.__str__() + "fps")
    frames_current_second += 1
    last_frame_time = current_frame_time
    delta *= 5
def handleInput():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            global run
            run = False
    keys = pygame.key.get_pressed()
    speed = 1.5
    turnspeed = 0.6
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
        camera.position[1] += speed * delta
    if keys[K_e]:
        camera.position[1] -= speed * delta
    if keys[K_r]:
        camera.rotation[0] += turnspeed * delta
    if keys[K_f]:
        camera.rotation[0] -= turnspeed * delta


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
    print(point_3D)
    d4_world_position = array([point_3D[0], point_3D[1], point_3D[2], 1])
    clipspace_position = projection_matrix @ view_matrix @ d4_world_position
    NDC_space = array([clipspace_position[0], clipspace_position[1], clipspace_position[2]]) / clipspace_position[3]
    x = ((NDC_space[0] + 1) / 2) * display.get_width()
    y = ((NDC_space[1] + 1) / 2) * display.get_height()

    screen_space = array([x, y, clipspace_position[2]])
    return screen_space


def update():
    global view_matrix
    view_matrix = createViewMatrix(camera)
    calcDelta()


def render():
    polygons = []
    display.fill([255, 255, 255])
    for model in models:
        for i in range(int(len(model.vertices) / 3)):
            v1 = fromWorldToScreen(model.vertices[i * 3 + 0])
            v2 = fromWorldToScreen(model.vertices[i * 3 + 1])
            v3 = fromWorldToScreen(model.vertices[i * 3 + 2])
            depth = -(v1[2] + v2[2] + v3[2]) / 3
            if v1[2] < near_plane or v2[2] < near_plane or v3[2] < near_plane:
                continue
            else:
                vertices = array([v1[0:2], v2[0:2], v3[0:2]])
                colors = model.colors[i]
                polygons.append(Polygon_2D(vertices, colors, depth))

    polygons.sort(key=lambda x: x.depth)

    for polygon in polygons:
        pygame.draw.polygon(display, polygon.color, polygon.vertices)

    pygame.display.update()
    pygame.display.flip()


def program():
    for i in range(2):
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
                          p[3], p[0], p[1],
                          p[3], p[1], p[2],
                          p[7], p[5], p[4],
                          p[7], p[6], p[5]])

        print(vertices)
        color = array([255.0, 0.0, 0.0])

        colors = array([color,
                        color,
                        color,
                        color,
                        color,
                        color,
                        color,
                        color,
                        color,
                        color])

        cube = Model(position, vertices, colors)
        models.append(cube)

    while run:
        handleInput()
        update()
        render()
    pygame.quit()


program()
