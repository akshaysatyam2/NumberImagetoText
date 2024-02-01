from PIL import Image, ImageDraw, ImageFont
import os
import inflect

# Create a directory to store the images
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Create an inflect engine
p = inflect.engine()

# Loop from 1 to 100
for i in range(1, 101):
    # Create a blank image
    img = Image.new('RGB', (100, 100), color='white')
    draw = ImageDraw.Draw(img)

    # Use a built-in font
    font = ImageFont.load_default()

    # Convert the number to words
    text = p.number_to_words(i).capitalize()

    # Position the text in the center
    text_position = ((100 - 82) // 2, (100 - 12) // 2)

    # Draw the text on the image
    draw.text(text_position, text, font=font, fill='black')

    # Save the image
    img.save(f'dataset/{str(i)}/capitalize_{i}.png')
