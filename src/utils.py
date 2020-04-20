

def circle_touches_rect(x: float, y: float, radius: float, rx: float, ry: float, w: float, h: float):
    cx = rx + w/2
    cy = ry + h/2

    dx = max(abs(x - cx) - w / 2, 0)
    dy = max(abs(y - cy) - h / 2, 0)
    d = dx * dx + dy * dy
    return d <= radius

