"""Visual explanations: Muse Image when configured, otherwise a generated SVG concept map."""
import html
import math

from .models import generate_image

PALETTE = ["#4f7cff", "#e5833a", "#2fa57a", "#b455d6", "#d6486b", "#3aa6c4"]


def concept_map_svg(bridges, paths, title):
    edges = [(p["from"], p["to"], p["relation"]) for p in paths]
    nodes = []
    for a, b, _ in edges:
        for n in (a, b):
            if n not in nodes:
                nodes.append(n)
    for br in bridges:
        if br["entity"] not in nodes:
            nodes.append(br["entity"])
    nodes = nodes[:12]
    if not nodes:
        return None
    W, H, cx, cy, r = 720, 460, 360, 235, 170
    pos = {n: (cx + r * math.cos(2 * math.pi * i / len(nodes) - math.pi / 2),
               cy + r * 0.78 * math.sin(2 * math.pi * i / len(nodes) - math.pi / 2)) for i, n in enumerate(nodes)}
    bridge_names = {b["entity"] for b in bridges}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif">',
           f'<text x="16" y="26" font-size="15" font-weight="600" fill="#888">{html.escape(title[:70])}</text>']
    for a, b, rel in edges:
        if a in pos and b in pos:
            out.append(f'<line x1="{pos[a][0]:.0f}" y1="{pos[a][1]:.0f}" x2="{pos[b][0]:.0f}" y2="{pos[b][1]:.0f}" '
                       f'stroke="#99a" stroke-opacity=".6" stroke-width="1.5"/>')
    for i, n in enumerate(nodes):
        x, y = pos[n]
        col = "#e5833a" if n in bridge_names else PALETTE[0]
        w = max(60, len(n) * 7.2 + 18)
        out.append(f'<rect x="{x - w / 2:.0f}" y="{y - 14:.0f}" width="{w:.0f}" height="28" rx="14" fill="{col}" fill-opacity=".18" '
                   f'stroke="{col}" stroke-width="1.5"/><text x="{x:.0f}" y="{y + 4:.0f}" text-anchor="middle" font-size="12" '
                   f'fill="#667">{html.escape(n)}</text>')
    out.append('<text x="16" y="448" font-size="11" fill="#999">Orange = concept shared across documents</text></svg>')
    return "".join(out)


def explain_visually(question, answer, bridges, paths):
    img = generate_image(f"Clear explanatory diagram answering: {question}\nKey points: {answer[:600]}")
    if img:
        return {"kind": "image", "mime": img["mime"], "data": img["data"], "source": "Muse Image"}
    svg = concept_map_svg(bridges, paths, question)
    return {"kind": "svg", "svg": svg, "source": "local concept map"} if svg else None
