import numpy as np
from PIL import Image
import os

# Create test_data directory
os.makedirs('test_data', exist_ok=True)

# Create a simple image
img = Image.new('RGB', (640, 480), color='red')
img.save('test_data/test_image.png')

print('Created test image: test_data/test_image.png')
