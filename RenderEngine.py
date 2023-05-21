import math

from numpy import array
import pygame
import Noise
import MyMath
import time

far_plane = 20
near_plane = 0.1
width = 1800
height = 950
display = pygame.display.set_mode(size=(width, height))

delta = 1
last_frame_time = time.time()
last_second_time = time.time()
frames_current_second = 0
frames_last_second = 0


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
camera.position[0] = 9500
camera.position[1] = 6
camera.position[0] = 9500

view_matrix = MyMath.createViewMatrix(camera)
projection_matrix = MyMath.createProjectionMatrix(near_plane, far_plane, display.get_width(), display.get_height())


def update():
    global view_matrix
    view_matrix = MyMath.createViewMatrix(camera)
    calcDelta()


def render(models):
    polygons = []
    day_color = array([178, 255, 255])
    night_color = array([11, 16, 38])
    time_factor = ((math.sin(time.time())+1)*0.5)
    sky_color = MyMath.lerp(time_factor, day_color, night_color)


    display.fill(sky_color)

    for model in models:
        for i in range(int(len(model.transformed_vertices) / 3)):
            v1 = fromWorldToScreen(model.transformed_vertices[i * 3 + 0])
            v2 = fromWorldToScreen(model.transformed_vertices[i * 3 + 1])
            v3 = fromWorldToScreen(model.transformed_vertices[i * 3 + 2])
            depth = (v1[2] + v2[2] + v3[2]) / 3
            if v1[2] < near_plane or v2[2] < near_plane or v3[2] < near_plane:
                continue
            else:
                vertices = array([v1[0:2], v2[0:2], v3[0:2]])
                polygons.append(Polygon_2D(vertices, model.last_calculated_colors[i], depth))

    polygons.sort(key=lambda x: x.depth, reverse= True)

    for polygon in polygons:
        #inverterat y-axeln
        polygon.vertices[0][1] = height - polygon.vertices[0][1]
        polygon.vertices[1][1] = height - polygon.vertices[1][1]
        polygon.vertices[2][1] = height - polygon.vertices[2][1]

        fog = (far_plane-polygon.depth)/(far_plane-far_plane*0.5)
        fog = pygame.math.clamp(fog, 0, 1)
        color = MyMath.lerp(1-fog, polygon.color, sky_color)
        color[0] =pygame.math.clamp(color[0], 0, 255)
        color[1] =pygame.math.clamp(color[1], 0, 255)
        color[2] =pygame.math.clamp(color[2], 0, 255)

        pygame.draw.polygon(display, color, polygon.vertices)
        # pygame.gfxdraw.filled_polygon(display, polygon.vertices, polygon.color,)

    # drawMap(array([int(width / 2), int(height / 2)]), 600, array([camera.position[0], camera.position[2]]))
    # pygame.display.update()
    pygame.display.flip()


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


def drawMap(position_screen, resolution, position_world):
    r = int(resolution / 2)
    zoom_out = 0.25
    for x in range(-r, r, 1):
        for y in range(-r, r, 1):
            height = Noise.noiseFunction(x * zoom_out + position_world[0], y * zoom_out + position_world[1])
            color = array([0, 255, 0])
            if height <= 0:
                color = array([0, 0, 255])
            if height >= 1.2:
                color = array([255, 255, 255])

            color = color * ((height / 10) + 0.5)
            color[0] = pygame.math.clamp(color[0], 0, 255)
            color[1] = pygame.math.clamp(color[1], 0, 255)
            color[2] = pygame.math.clamp(color[2], 0, 255)
            display.set_at((x + position_screen[0], y + position_screen[1]), color)


def fromWorldToScreen(point_3D):
    clipspace_position = projection_matrix @ view_matrix @ point_3D
    if clipspace_position[3] == 0: clipspace_position[3] = 0.00001
    NDC_space = clipspace_position[0:2] / clipspace_position[3]
    x = ((NDC_space[0] + 1) / 2) * display.get_width()
    y = ((NDC_space[1] + 1) / 2) * display.get_height()
    screen_space = array([x, y, clipspace_position[2]])
    return screen_space
