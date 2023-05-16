import numpy as np
import pygame
import time
from pygame.locals import QUIT
from pygame.math import clamp
from numpy import array
from numpy import cos
from numpy import sin


run = True
width = 800
height = 800
display = pygame.display.set_mode((width, height))


def handleInput():
    for event in pygame.event.get():
        if event.type == QUIT:
            global run
            run = False


def update():
    return


def render():
    display.fill([255, 255, 255])

    vertices = array([[400.0, 400.0], [600.0, 650.0], [500.0, 400.0]])
    vertices[2] += sin(time.time())*100
    color = array([255, 0, 0])
    color[2] = 155+cos(time.time())*100
    color[2] = clamp(color[2], 0 , 255)
    pygame.draw.polygon(display, color, vertices)

    pygame.display.update()
    pygame.display.flip()

def program():
    while run:
        handleInput()
        update()
        render()
    pygame.quit()

program()