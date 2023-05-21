import pygame
import time
import pygame.gfxdraw
import RenderEngine
from MyMath import *
import Models
from pygame.locals import *  ## allt för att hämta konstanter till varje tangent

run = True


class Player:
    def __init__(self):
        self.position = array([0, 0, 0])
        self.velocity = array([0, 0, 0])

        self.model = Models.generateAirplane()
        self.model.position = RenderEngine.camera.position + array([0, 3, 3])
        models_needing_update.append(self.model)
        models.append(self.model)


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


def handleInput():
    delta = RenderEngine.delta
    speed = 8.0
    turnspeed = 2.5
    camera = RenderEngine.camera
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


def hasChunk(x, y):
    for chunk in chunks:
        if chunk.x == x and chunk.y == y: return True
    return False


def update():
    RenderEngine.update()
    light_direction = array([0.9, -0.7, .9])
    light_direction = light_direction / np.linalg.norm(light_direction)

    ##TA BORT ENDAST FULT TEST
    models[0].rotation[0] = sin(time.time() * 0.6) * 0.3
    models[0].rotation[1] = sin(time.time() * 0.3) * 0.9
    models[0].rotation[2] = sin(time.time() * 0.7) * 0.7
    models_needing_update.append(models[0])

    load_chunk_world_distance = 8
    distance = load_chunk_world_distance
    camera = RenderEngine.camera
    x = camera.position[0] - distance * sin(-camera.rotation[1])
    z = camera.position[2] - distance * cos(-camera.rotation[1])
    target_point = array([x / Chunk.chunk_size, z / Chunk.chunk_size])

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
    player = Player()
    while run:
        handleInput()
        update()
        render()
    pygame.quit()


program()
