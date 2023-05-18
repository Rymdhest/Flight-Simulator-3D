import math

def interpolate(a, b, x):
    ft = x * math.pi
    f = (1 - math.cos(ft)) * 0.5
    return a * (1 - f) + b * f


def noise(x, y):
    n = x + y * 57
    n = (n << 13) ^ n
    return (1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0)


def smooth_noise(x, y):
    corners = (noise(x - 1, y - 1) + noise(x + 1, y - 1) +
               noise(x - 1, y + 1) + noise(x + 1, y + 1)) / 16
    sides = (noise(x - 1, y) + noise(x + 1, y) +
             noise(x, y - 1) + noise(x, y + 1)) / 8
    center = noise(x, y) / 4
    return corners + sides + center


def interpolated_noise(x, y):
    integer_X = int(x)
    fractional_X = x - integer_X

    integer_Y = int(y)
    fractional_Y = y - integer_Y

    v1 = smooth_noise(integer_X, integer_Y)
    v2 = smooth_noise(integer_X + 1, integer_Y)
    v3 = smooth_noise(integer_X, integer_Y + 1)
    v4 = smooth_noise(integer_X + 1, integer_Y + 1)

    i1 = interpolate(v1, v2, fractional_X)
    i2 = interpolate(v3, v4, fractional_X)

    return interpolate(i1, i2, fractional_Y)


def perlin_noise(x, y, persistence, octaves):
    total = 0
    frequency = 1
    amplitude = 1
    max_value = 0

    for _ in range(octaves):
        total += interpolated_noise(x * frequency, y * frequency) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= 2

    return total / max_value