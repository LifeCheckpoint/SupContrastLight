import os
from PIL import Image, UnidentifiedImageError
from pathlib import Path

def process_images(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
    
    hei, wid = 384, 384

    for img_path in input_folder.iterdir():
        if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
            try:
                with Image.open(img_path) as img:
                    # Convert image to RGB mode
                    img = img.convert('RGB')

                    # Resize image
                    width, height = img.size
                    if width < height:
                        new_width = wid
                        new_height = int(height * (wid / width))
                    else:
                        new_height = hei
                        new_width = int(width * (hei / height))

                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                    # Crop the center
                    left = (new_width - wid) / 2
                    top = (new_height - hei) / 2
                    right = (new_width + wid) / 2
                    bottom = (new_height + hei) / 2
                    img = img.crop((left, top, right, bottom))

                    # Save the image to the output folder
                    output_path = output_folder / img_path.name
                    img.save(output_path)

            except UnidentifiedImageError:
                print(f"Cannot identify image file {img_path}")
            except Exception as e:
                print(f"An error occurred with file {img_path}: {e}")

input_folder = 'D:/Dataset/pic/genhon'
output_folder = 'D:/Dataset/pic/genhon_c'
process_images(input_folder, output_folder)
