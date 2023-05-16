import pygame
import numpy
from pygame.locals import QUIT

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
    return


def program():
    while run:
        handleInput()
        update()
        render()
    pygame.quit()

program()