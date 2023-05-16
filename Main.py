import pygame
import time
from math import pi
import numpy
from pygame.math import clamp
from numpy import array
from numpy import cos
from numpy import sin
from pygame.locals import * ## allt för att hämta konstanter till varje tangent
run = True
width = 800
height = 800
display = pygame.display.set_mode((width, height))


class Model:
    def __init__(self, position, vertices, colors=array([0.0, 0.0, 0.0]), rotation=array([0.0, 0.0, 0.0]),
                 scale=array([1.0, 1.0, 1.0])):
        self.position = position
        self.scale = scale
        self.vertices = vertices
        self.colors = colors
        self.rotation = rotation


class Camera:
    def __init__(self):
        self.position = array([0.0, 0.0, 0.0])
        self.rotation = array([0.0, 0, 0.0, 0.0])


camera = Camera()
camera.position[2] = 2
models = []


def handleInput():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            global run
            run = False
    keys = pygame.key.get_pressed()
    speed = 1.5
    turnspeed = 0.6
    delta = 0.01
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
    far_plane = 100
    near_plane = 1
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
    d4_world_position = array([point_3D[0], point_3D[1], point_3D[2], 1])
    clipspace_position = projection_matrix @ view_matrix @ d4_world_position
    NDC_space = array([clipspace_position[0], clipspace_position[1], clipspace_position[2]]) / clipspace_position[3]
    x = ((NDC_space[0] + 1) / 2) * display.get_width()
    y = ((NDC_space[1] + 1) / 2) * display.get_height()

    screen_space = array([x, y])
    return screen_space


def update():
    global view_matrix
    view_matrix = createViewMatrix(camera)


def render():
    display.fill([255, 255, 255])

    vertices = array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, -2.0, 0.0]])
    color = array([255, 0, 0])
    p1_2D = fromWorldToScreen(vertices[0])
    p2_2D = fromWorldToScreen(vertices[1])
    p3_2D = fromWorldToScreen(vertices[2])
    pygame.draw.polygon(display, color, array([p1_2D, p2_2D, p3_2D]))

    vertices = array([[3.0, 0.0, 0.0], [3.0, 0.0, -3.0], [3.0, -2.0, -3.0]])
    color = array([255, 255, 0])
    p1_2D = fromWorldToScreen(vertices[0])
    p2_2D = fromWorldToScreen(vertices[1])
    p3_2D = fromWorldToScreen(vertices[2])
    pygame.draw.polygon(display, color, array([p1_2D, p2_2D, p3_2D]))

    pygame.display.update()
    pygame.display.flip()


def program():
    while run:
        handleInput()
        update()
        render()
    pygame.quit()


program()
