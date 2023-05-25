import math
import random

import numpy as np
import numpy.linalg
import pygame
import time
import MyMath
import Noise
import RenderEngine
from MyMath import *
import Models
from pygame.locals import *  ## allt för att hämta konstanter till varje tangent

run = True


class Player:
    def __init__(self):

        self.model = Models.generateAirplane()
        self.model.position[1] = 6
        self.model.position[0] = 5645
        self.model.position[2] = 4567868
        models_needing_update.append(self.model)
        models.append(self.model)

        self.momentum = array([0.0, 0.0, 0.0])

        self.crashed = False
        self.crash_time = 0


    def increaseForwardMomentum(self, direction):
        thrust_power = 0.035
        amount = thrust_power * direction * RenderEngine.delta
        forward_vector = array([0.0, 0.0, amount, 1.0])
        forward_vector = forward_vector @ self.model.rotation_matrix
        forward_vector = np.delete(forward_vector, -1)
        self.momentum += forward_vector

    def crash(self):
        self.momentum = self.momentum*0.0
        self.crashed = True


    def update(self):
        if self.crashed:
            self.crash_time += RenderEngine.delta*70.0
            random.seed(1337)
            explode_amount = (1 / (self.crash_time ** 2)) * 20
            for i in range(int(len(self.model.vertices) / 3)):

                rand1 = random.uniform(-1.0, 1.0) * explode_amount
                rand2 = random.uniform(-1.0, 1.0) * explode_amount
                rand3 = random.uniform(-1.0, 1.0) * explode_amount

                self.model.vertices[i * 3 + 0] = self.model.vertices[i * 3 + 0] + array([rand1, rand2, rand3])
                self.model.vertices[i * 3 + 1] = self.model.vertices[i * 3 + 1] + array([rand1, rand2, rand3])
                self.model.vertices[i * 3 + 2] = self.model.vertices[i * 3 + 2] + array([rand1, rand2, rand3])

            models_needing_update.append(self.model)
            return
        delta = RenderEngine.delta
        gravity = -0.09 * delta
        air_pressure = ((-self.model.position[1] ** 3) * 0.001 + 1)
        speed = np.linalg.norm(self.momentum)
        lift_power = air_pressure * 0.45
        lift_vector = array([0.0, speed * lift_power * delta, 0.0, 1.0])
        lift_vector = lift_vector @ self.model.rotation_matrix
        lift_vector = np.delete(lift_vector, -1)




        self.momentum = self.momentum + lift_vector
        print(self.momentum[1])

        ground_height = Noise.noiseFunction(self.model.position[0], self.model.position[2])
        if self.model.position[1] < ground_height + 0.2:
            self.model.position[1] = ground_height + 0.2

            if self.momentum[1] <= -0.05:
                self.crash()
            self.momentum[1] = 0
            self.momentum = self.momentum - self.momentum * delta * 0.35
            plane_length = 2.0
            back =  np.delete( (np.array([0.0 ,0.0, -plane_length, 1.0])) @ self.model.rotation_matrix, -1)
            back_height = Noise.noiseFunction(self.model.position[0]+ back[0], self.model.position[2]+ back[2])
            back[1] = back_height
            p1 = player.model.position
            p2 = back+player.model.position
            angle = -np.arcsin((p2[1]-p1[1]) / (np.linalg.norm(p2-p1)))*1.0
            if angle > math.pi:
                angle = angle-math.pi*2

            #player.model.rotate([angle,0,0])

            #print(ground_height)
            #print(back_height)

        else:
            self.momentum[1] = self.momentum[1] + gravity
            self.momentum = self.momentum - self.momentum * delta * 0.1

        all_momentun_forward =  np.delete( (np.array([0.0 ,0.0, 1.0, 1.0])*speed) @ self.model.rotation_matrix, -1)
        air_grip = (1-1/(30.0*delta*(speed**2)+1))


        dy = -self.momentum[1]
        self.momentum = self.momentum+self.momentum*dy*0.045
        self.momentum = self.momentum*(1-air_grip) + all_momentun_forward*(air_grip)

        self.model.position = self.model.position + self.momentum*delta*10


class Chunk:
    chunk_size = 2

    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
        self.center_world_pos = array(
            [self.x * self.chunk_size + self.chunk_size / 2, self.y * self.chunk_size + self.chunk_size / 2])
        model = Models.generateTerrainChunkModel(self.x * self.chunk_size, self.y * self.chunk_size,
                                                 self.chunk_size + 1)
        models_needing_update.append(model)
        self.model = model
        models.append(model)

    def cleanUp(self):
        models.remove(self.model)


models = []
models_needing_update = []
chunks = []
player = Player()
player.model.rotate([0, math.pi, 0])
mouse_down_coords = [0,0]
camera_offset_angle = array([0.0,0.0])

def handleInput():
    delta = RenderEngine.delta
    global mouse_down_coords
    turnspeed = 2.5
    global camera_offset_angle
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            global run
            run = False

        if event.type == MOUSEBUTTONDOWN:
            pygame.mouse.set_visible(False)
            mouse_down_coords = pygame.mouse.get_pos()

        if event.type == MOUSEBUTTONUP:
            pygame.mouse.set_visible(True)
            pygame.mouse.set_pos(mouse_down_coords)
            camera_offset_angle[0] = 0
            camera_offset_angle[1] = 0
        if event.type == MOUSEMOTION and pygame.mouse.get_pressed()[0] and not player.crashed:
            movement = pygame.mouse.get_rel()

            player.model.rotate(array([0, 0, -movement[0] * delta * 0.05]))
            player.model.rotate(array([movement[1] * delta * 0.05, 0, 0]))

            pygame.mouse.set_pos([RenderEngine.width / 2, RenderEngine.height / 2])
        elif event.type == MOUSEMOTION and pygame.mouse.get_pressed()[2]:
            movement = pygame.mouse.get_rel()

            camera_offset_angle[0] += movement[0]*delta*0.05
            camera_offset_angle[1] -= movement[1]*delta*0.05
            print(camera_offset_angle)
        else:
            pygame.mouse.get_rel()
    keys = pygame.key.get_pressed()

    if not player.crashed:
        if keys[K_w]:
            player.increaseForwardMomentum(1)
        if keys[K_s]:
            player.increaseForwardMomentum(-1)
        if keys[K_a]:
            player.model.rotate(array([0, -turnspeed * delta, 0]))
        if keys[K_d]:
            player.model.rotate(array([0, turnspeed * delta, 0]))
        if keys[K_r]:
            player.model.rotate(array([turnspeed * delta, 0, 0]))
        if keys[K_f]:
            player.model.rotate(array([-turnspeed * delta, 0, 0]))
        if keys[K_q]:
            player.model.rotate(array([0, 0, turnspeed * delta]))
        if keys[K_e]:
            player.model.rotate(array([0, 0, -turnspeed * delta]))
    if keys[K_KP_PLUS]:
        RenderEngine.map_zoom_out -= 0.5*delta
        print(RenderEngine.map_zoom_out)
    if keys[K_KP_MINUS]:
        RenderEngine.map_zoom_out += 0.5*delta
        print(RenderEngine.map_zoom_out)


def hasChunk(x, y):
    for chunk in chunks:
        if chunk.x == x and chunk.y == y: return True
    return False


def update():
    player.update()
    global camera_offset_angle
    camera = RenderEngine.camera
    camera_offset = array([0, 1.12, -5])

    camera.rotation_matrix = np.identity(4)
    camera.rotation_matrix = camera.rotation_matrix @ MyMath.createRotationMatrix(array([math.pi / 64, math.pi, 0, 1.0]))
    camera.rotation_matrix = camera.rotation_matrix @ MyMath.createRotationMatrix(array([camera_offset_angle[1], camera_offset_angle[0], 0]))

    camera.rotation_matrix = camera.rotation_matrix @ np.copy(player.model.rotation_matrix)

    camera_offset = np.append(camera_offset, 1.0)
    camera_offset = camera_offset  @ MyMath.createRotationMatrix(array([camera_offset_angle[1], camera_offset_angle[0], 0]))@ player.model.rotation_matrix
    camera_offset = np.delete(camera_offset, -1)

    camera.position = player.model.position + camera_offset
    offset = 1
    height_at_camera = Noise.noiseFunction(camera.position[0], camera.position[2])
    if camera.position[1] <= height_at_camera+offset:
        camera.position[1] = height_at_camera+offset

    light_direction = array([0.9, -0.7, .9])
    light_direction = light_direction / np.linalg.norm(light_direction)

    models_needing_update.append(player.model)

    load_chunk_world_distance = 8
    distance = load_chunk_world_distance

    forward = array([0, 0, -14, 1.0]) @ camera.rotation_matrix
    x = camera.position[0] + forward[0]
    z = camera.position[2] + forward[2]

    target_point = array([x / Chunk.chunk_size, z / Chunk.chunk_size])
    RenderEngine.update(player)

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

    for model in models_needing_update:
        transformed_vertices = np.append(model.vertices, np.ones((len(model.vertices), 1)), axis=1)

        transformed_vertices = transformed_vertices @ createScaleMatrix(model.scale)
        transformed_vertices = transformed_vertices @ model.rotation_matrix
        transformed_vertices = transformed_vertices + (np.append(model.position, 1.0))

        transformed_normals = np.append(model.normals, np.ones((len(model.normals), 1)), axis=1)
        transformed_normals = transformed_normals @ model.rotation_matrix
        for i in range(int(len(transformed_vertices) / 3)):
            lighting = np.dot(transformed_normals[i][0:3], light_direction)
            lighting = max(lighting, 0.0)*0.6 + 0.4
            colors = model.colors[i] * lighting
            colors = np.clip(colors, 0, 255)
            model.last_calculated_colors[i] = colors
            model.transformed_vertices = transformed_vertices
    models_needing_update.clear()


def render():
    RenderEngine.render(models, player)


def program():
    # RenderEngine.drawMap(array([int(RenderEngine.width / 2), int(RenderEngine.height / 2)]), 100, array([player.model.position[0], player.model.position[2]]))

    while run:
        handleInput()
        update()
        render()
    pygame.quit()


program()
