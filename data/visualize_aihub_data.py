import base64
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import os
import textwrap

def decode_base64_image(base64_str):
    """Decode a base64-encoded image."""
    try:
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def generate_mask_from_pts_string(pts_string, image_size):
    """Generate a binary mask image from the polygon points string."""
    try:
        polygons = []
        for poly_str in pts_string.strip().split(';'):
            if not poly_str:
                continue
            coords = list(map(float, poly_str.strip().split(',')))
            # Group the coordinates into (x, y) pairs
            points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            polygons.append(points)

        # Create a blank mask image
        mask = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(mask)
        # Draw the polygons onto the mask
        for polygon in polygons:
            draw.polygon(polygon, outline=1, fill=1)
        return mask
    except Exception as e:
        print(f"Error generating mask from pts_string: {e}")
        return None

def overlay_mask_on_image(image, mask, alpha=0.5):
    """Overlay a binary mask on an image."""
    # Resize mask to match image size if necessary
    if image.size != mask.size:
        mask = mask.resize(image.size, resample=Image.NEAREST)

    # Convert images to numpy arrays
    image_np = np.array(image).astype(np.float32)
    mask_np = np.array(mask)

    # Ensure mask is binary
    mask_np = (mask_np > 0)

    # Create the red color array
    red_color = np.array([255, 0, 0], dtype=np.float32)

    # Apply the overlay where mask is True
    overlay_np = image_np.copy()

    # Broadcasting the operation over the masked pixels
    overlay_np[mask_np] = (1 - alpha) * overlay_np[mask_np] + alpha * red_color

    # Convert back to uint8
    overlay_np = overlay_np.astype(np.uint8)

    # Convert back to PIL Image
    blended = Image.fromarray(overlay_np)
    return blended

def add_referring_text(image, text):
    """Add referring text to the image with text wrapping."""
    draw = ImageDraw.Draw(image)
    font_size = max(15, image.size[0] // 50)
    try:
        # Update the font path to point to the Korean font
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print("Korean font not found. Text will not be added.")
        return image

    text_color = (255, 255, 255)  # White color
    outline_color = (0, 0, 0)     # Black outline

    # Determine the maximum width for the text
    max_width = image.size[0] - 20  # 10 pixels padding on each side

    # Wrap the text
    lines = textwrap.wrap(text, width=40)  # Adjust width as needed

    y_text = 10
    for line in lines:
        line_width, line_height = font.getsize(line)
        x_text = 10

        # Draw outline
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    draw.text((x_text+dx, y_text+dy), line, font=font, fill=outline_color)

        # Draw text
        draw.text((x_text, y_text), line, font=font, fill=text_color)
        y_text += line_height

    return image

def visualize_tsv_entry(entry, save_dir):
    """Visualize a single TSV entry and save the image."""
    # Entry is a line from the TSV file
    fields = entry.strip().split('\t')
    if len(fields) != 8:
        print("Incorrect number of fields:", len(fields))
        return
    uniq_id, image_id, sent, box_string, pts_string, img_base64, annot_base64, pts_string_interpolated = fields

    # Decode images
    img = decode_base64_image(img_base64)
    if img is None:
        print("Failed to decode image.")
        return

    # Generate mask from polygon points
    mask = generate_mask_from_pts_string(pts_string, img.size)
    if mask is None:
        print("Failed to generate mask from polygon.")
        return

    # Overlay mask on image
    blended = overlay_mask_on_image(img, mask)
    if blended is None:
        print("Failed to overlay mask on image.")
        return

    # Add referring text to the image
    blended_with_text = add_referring_text(blended, sent)

    # Save the image
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{uniq_id}.png"
    filepath = os.path.join(save_dir, filename)
    blended_with_text.save(filepath)

    # Optionally, display the image
    # plt.figure(figsize=(10, 10))
    # plt.imshow(blended_with_text)
    # plt.axis('off')
    # plt.show()

    print(f"Saved image {filepath}")

def main():
    # Update this path to point to your TSV file
    tsv_file = 'datasets/finetune/aihub_indoor_bbox_fix/aihub_indoor_train.tsv'
    # tsv_file = 'datasets/finetune/aihub_indoor_bbox_fix/aihub_indoor_val.tsv'

    # Directory where the images will be saved
    save_dir = 'visualizations_train'

    with open(tsv_file, 'r') as f:
        lines = f.readlines()

    num_entries = len(lines)
    print(f"Total entries: {num_entries}")

    while True:
        index = input(f"Enter entry index (0 to {num_entries - 1}), 'all' to process all entries, or 'q' to quit: ")
        if index.lower() == 'q':
            break
        elif index.lower() == 'all':
            for idx in range(num_entries):
                entry = lines[idx]
                visualize_tsv_entry(entry, save_dir)
            print(f"All images have been saved to {save_dir}")
            break
        else:
            try:
                index = int(index)
                if 0 <= index < num_entries:
                    entry = lines[index]
                    visualize_tsv_entry(entry, save_dir)
                else:
                    print("Index out of range.")
            except ValueError:
                print("Invalid input.")

if __name__ == '__main__':
    main()
