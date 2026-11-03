import sys
from PIL import Image, ImageDraw

size, path = sys.argv[1], sys.argv[2]
width, height = (int(v) for v in size.split("x"))
image = Image.new("RGB", (width, height))
pixels = image.load()
for y in range(height):
    for x in range(width):
        pixels[x, y] = (255 * x // width, 255 * y // height, 128)
draw = ImageDraw.Draw(image)
draw.ellipse((width // 4, height // 4, width * 3 // 4, height * 3 // 4), fill=(230, 40, 40))
draw.rectangle((width // 10, height // 10, width // 4, height // 3), fill=(30, 30, 200))
image.save(path)
