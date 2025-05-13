from PIL import Image

# Open the JPEG image
image = Image.open("./data/Dataset/0/IMG_1118.JPG")

# Convert to RGB (in case it's grayscale or CMYK)
image = image.convert("RGB")

# Get width and height
width, height = image.size

# Access pixel values
pixels = []
for y in range(height):
    row = []
    for x in range(width):
        r, g, b = image.getpixel((x, y))  # returns a tuple (R, G, B)
        row.append((r, g, b))
    pixels.append(row)

# Print the first 5 pixels of the first row
print("First 5 pixels in the first row:", pixels[0][:5])

