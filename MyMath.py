import math
import numpy as np
from numpy import array
from numpy import cos
from numpy import sin


def lerp(amount, left, right):
    return (1.0 - amount) * left + amount * right


def barryCentric(p1, p2, p3, pos):
    det = (p2[2] - p3[2]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[2] - p3[2])
    l1 = ((p2[2] - p3[2]) * (pos[0] - p3[0]) + (p3[0] - p2[0]) * (pos[1] - p3[2])) / det
    l2 = ((p3[2] - p1[2]) * (pos[0] - p3[0]) + (p1[0] - p3[0]) * (pos[1] - p3[2])) / det
    l3 = 1.0 - l1 - l2
    return l1 * p1[1] + l2 * p2[1] + l3 * p3[1]


def createViewMatrix(camera):
    matrix = np.identity(4)
    matrix = matrix @ camera.rotation_matrix
    matrix = matrix @ createTranslateMatrix(-camera.position)
    return matrix


def createProjectionMatrix(near_plane, far_plane, width, height, field_of_view):
    aspect_ratio = width / height
    y_scale = 1 / np.arctan(field_of_view / 2)
    x_scale = y_scale / aspect_ratio
    frustum_length = far_plane - near_plane
    A = -((far_plane + near_plane) / frustum_length)
    B = -((2 * far_plane * near_plane) / frustum_length)
    projection_matrix = array([[x_scale, 0, 0, 0],
                               [0, y_scale, 0, 0],
                               [0, 0, A, B],
                               [0, 0, -1, 0]])
    return projection_matrix


def createScaleMatrix(scale):
    scale_matrix = np.identity(4)
    scale_matrix[0, 0] = scale[0]
    scale_matrix[1, 1] = scale[1]
    scale_matrix[2, 2] = scale[2]
    return scale_matrix


def createTranslateMatrix(translation):
    translate_matrix = np.identity(4)
    translate_matrix[0:3, 3] = translation
    return translate_matrix


def createRotationMatrix(rotation):
    rotation_x_matrix = array([[1, 0, 0, 0],
                               [0, cos(rotation[0]), -sin(rotation[0]), 0],
                               [0, sin(rotation[0]), cos(rotation[0]), 0],
                               [0, 0, 0, 0]])
    rotation_y_matrix = array([[cos(rotation[1]), 0, sin(rotation[1]), 0],
                               [0, 1, 0, 0],
                               [-sin(rotation[1]), 0, cos(rotation[1]), 0],
                               [0, 0, 0, 0]])
    rotation_z_matrix = array([[cos(rotation[2]), -sin(rotation[2]), 0, 0],
                               [sin(rotation[2]), cos(rotation[2]), 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 0]])
    return rotation_x_matrix @ rotation_y_matrix @ rotation_z_matrix


def getDistanceBetween_2D(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)

def getDistanceBetween_3D(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


