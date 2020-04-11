

def circle_touches_rect(x: float, y: float, radius: float, rx: float, ry: float, w: float, h: float):
    dx = max(abs(x - rx) - w / 2, 0)
    dy = max(abs(y - ry) - h / 2, 0)
    d = dx * dx + dy * dy
    return d <= radius

