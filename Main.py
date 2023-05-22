import pygame
import time
import pygame.gfxdraw

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
        models_needing_update.append(self.model)
        models.append(self.model)

        self.momentum = array([0.0, 0.0, 0.0])

    def increaseForwardMomentum(self, direction):
        speed = 0.75
        amount = speed*direction*RenderEngine.delta
        forward_vector = array([0.0, 0.0, amount, 1.0])
        forward_vector = forward_vector @ createRotationMatrix(self.model.rotation)
        forward_vector = np.delete(forward_vector, -1)
        self.momentum += forward_vector


    def update(self):
        delta = RenderEngine.delta
        gravity = -0.6 * delta
        lift_power = ((-self.model.position[1]**3)*0.001+1)*0.5
        lift_vector = array([0.0, self.momentum[2] * lift_power * delta, 0.0, 1.0])
        lift_vector = lift_vector @ createRotationMatrix(self.model.rotation)
        lift_vector = np.delete(lift_vector, -1)
        self.momentum = self.momentum+lift_vector

        self.model.position = self.model.position+self.momentum*RenderEngine.delta

        ground_height = Noise.noiseFunction(self.model.position[0], self.model.position[2])
        if self.model.position[1] < ground_height+0.2:
            self.model.position[1] = ground_height+0.2
            self.momentum = self.momentum - self.momentum*delta*0.3

            self.model.rotation[0] = 0
            self.model.rotation[2] = 0
        else:
            self.momentum[1] = self.momentum[1] + gravity
            self.momentum = self.momentum - self.momentum*delta*0.1

class Chunk:
    chunk_size = 4

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

def handleInput():
    delta = RenderEngine.delta

    turnspeed = 2.5
    camera = RenderEngine.camera
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            global run
            run = False
        if event.type == MOUSEMOTION:
            movement = pygame.mouse.get_rel()
            player.model.rotation[2] -= movement[0] * delta * 0.05
            player.model.rotation[0] += movement[1] * delta * 0.05
    keys = pygame.key.get_pressed()

    if keys[K_w]:
        player.increaseForwardMomentum(1)
    if keys[K_s]:
        player.increaseForwardMomentum(-1)
    if keys[K_a]:
        player.model.rotation[1] -= turnspeed * delta
    if keys[K_d]:
        player.model.rotation[1] += turnspeed * delta
    if keys[K_r]:
        player.model.rotation[0] += turnspeed * delta
    if keys[K_f]:
        player.model.rotation[0] -= turnspeed * delta
    if keys[K_q]:
        player.model.rotation[2] += turnspeed * delta
    if keys[K_e]:
        player.model.rotation[2] -= turnspeed * delta


def hasChunk(x, y):
    for chunk in chunks:
        if chunk.x == x and chunk.y == y: return True
    return False


def update():

    player.update()

    camera = RenderEngine.camera
    camera_offset = array([0, 2, -4])

    camera_pos = np.append(camera_offset, 1.0)
    camera_pos = camera_pos @ createRotationMatrix(player.model.rotation)
    camera_pos = np.delete(camera_pos, -1)

    camera.position = player.model.position+camera_pos

    camera.rotation[0] = -player.model.rotation[0]+math.pi/8
    camera.rotation[1] = player.model.rotation[1]-math.pi

    camera.rotation[2] = player.model.rotation[2]

    light_direction = array([0.9, -0.7, .9])
    light_direction = light_direction / np.linalg.norm(light_direction)

    models_needing_update.append(player.model)

    load_chunk_world_distance = 8
    distance = load_chunk_world_distance

    x = camera.position[0] - distance * sin(-camera.rotation[1])
    z = camera.position[2] - distance * cos(-camera.rotation[1])
    target_point = array([x / Chunk.chunk_size, z / Chunk.chunk_size])
    RenderEngine.update()

    for model in models_needing_update:
        transformed_vertices = np.append(model.vertices, np.ones((len(model.vertices), 1)), axis=1)
        transformed_vertices = transformed_vertices @ createRotationMatrix(model.rotation)
        transformed_vertices = transformed_vertices @ createScaleMatrix(model.scale)
        transformed_vertices = transformed_vertices + (np.append(model.position, 1.0))

        transformed_normals = np.append(model.normals, np.ones((len(model.normals), 1)), axis=1)
        transformed_normals = transformed_normals @ createRotationMatrix(model.rotation)
        for i in range(int(len(transformed_vertices) / 3)):
            lighting = np.dot(transformed_normals[i][0:3], light_direction)
            lighting = max(lighting, 0.0) + 0.4
            colors = model.colors[i] * lighting
            colors = np.clip(colors, 0, 255)
            model.last_calculated_colors[i] = colors
            model.transformed_vertices = transformed_vertices
    models_needing_update.clear()

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
    RenderEngine.render(models)


def program():

    while run:
        handleInput()
        update()
        render()
    pygame.quit()


program()
