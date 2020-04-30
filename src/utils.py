import numpy as np

def circle_touches_rect(x: float, y: float, radius: float, rx: float, ry: float, w: float, h: float):
    cx = rx + w/2
    cy = ry + h/2

    dx = max(abs(x - cx) - w / 2, 0)
    dy = max(abs(y - cy) - h / 2, 0)
    d = dx * dx + dy * dy
    return d <= radius

def circle_touches_circle(x: float, y: float, radius: float, x2: float, y2: float,radius2: float):
    if np.linalg.norm(np.array([x2 - x, y2 - y])) > radius + radius2:
        return False

    return True
