from PIL import Image

try:
    image_path = "temp_images/fd82305f83714327b66ef37f153038dd.jpg"
    img = Image.open(image_path)
    print("Image opened successfully!")
except Exception as e:
    print(f"Error opening image: {e}")
