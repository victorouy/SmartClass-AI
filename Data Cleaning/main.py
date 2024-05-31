import os
from PIL import Image, ImageEnhance, ImageStat

# def has_high_black_values(image, threshold=150):
#     stat = ImageStat.Stat(image)
#     min_pixel_value = stat.extrema[0][0]
#     return min_pixel_value < threshold

def get_brightness(image):
    stat = ImageStat.Stat(image)
    brightness = stat.mean[0]
    return brightness

def get_contrast(image):
    stat = ImageStat.Stat(image)
    stddev = stat.stddev[0]
    return stddev

def get_sharpness(image):
    image = image.convert("L")  # Convert to grayscale
    stat = ImageStat.Stat(image)
    return stat.stddev[0]

def has_high_brightness(image, threshold=150):
    brightness = get_brightness(image)
    return brightness > threshold, brightness

def has_low_brightness(image, threshold=100):
    brightness = get_brightness(image)
    return brightness < threshold, brightness

def has_high_contrast(image, threshold=175):
    contrast = get_contrast(image)
    return contrast > threshold, contrast

def has_low_contrast(image, threshold=60):
    contrast = get_contrast(image)
    return contrast < threshold, contrast

def has_high_sharpness(image, threshold=150):
    sharpness = get_sharpness(image)
    return sharpness > threshold, sharpness

def has_low_sharpness(image, threshold=50):
    sharpness = get_sharpness(image)
    return sharpness < threshold, sharpness


def clean_image(image_path, size=(48, 48)):
    image = Image.open(image_path).convert('L')

    # Resize image
    image = image.resize(size)

    high_brightness, brightness = has_high_brightness(image)
    low_brightness, brightness = has_low_brightness(image)
    if high_brightness:
        # Adjust brightness if it is high
        adjustment_factor = (brightness - 150) / 150
        image = ImageEnhance.Brightness(image).enhance(1 - adjustment_factor)
    elif low_brightness:
        # Adjust brightness if it is low
        adjustment_factor = (100 - brightness) / 100
        image = ImageEnhance.Brightness(image).enhance(1 + adjustment_factor)

    high_contrast, contrast = has_high_contrast(image)
    low_contrast, contrast = has_low_contrast(image)
    if high_contrast:
        # Adjust contrast if it is high
        adjustment_factor = (contrast - 175) / 175
        image = ImageEnhance.Contrast(image).enhance(1 - adjustment_factor)
    elif low_contrast:
        # Adjust contrast if it is low
        adjustment_factor = (60 - contrast) / 60
        image = ImageEnhance.Contrast(image).enhance(1 + adjustment_factor)

    # Adjust sharpness
    high_sharpness, sharpness = has_high_sharpness(image)
    low_sharpness, sharpness = has_low_sharpness(image)
    if high_sharpness:
        adjustment_factor = (sharpness - 150) / 150
        image = ImageEnhance.Sharpness(image).enhance(1 - adjustment_factor)
    elif low_sharpness:
        adjustment_factor = (50 - sharpness) / 50
        image = ImageEnhance.Sharpness(image).enhance(1 + adjustment_factor)

    return image


def clean_dataset(dataset_dir, size=(48, 48)):
    print('Started Cleaning...')
    for root, dirs, files in os.walk(dataset_dir):
        for dir_name in dirs:
            count = 0
            full_dir_path = os.path.join(root, dir_name)
            parent_folder = os.path.basename(root)
            for file in os.listdir(full_dir_path):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(full_dir_path, file)
                    cleaned_image = clean_image(image_path, size)

                    # Get the original file extension
                    file_extension = os.path.splitext(file)[1]

                    # Create new file name and ensure it's unique
                    while True:
                        new_file_name = f"{parent_folder}-{dir_name}_{count:04d}{file_extension}"
                        new_file_path = os.path.join(full_dir_path, new_file_name)
                        if not os.path.exists(new_file_path):
                            break
                        count += 1

                    # Save the cleaned image with the new name
                    cleaned_image.save(new_file_path)

                    # Remove the old file
                    os.remove(image_path)

                    count += 1
    print('Completed Cleaning.')


dataset_dir = 'dataset'  # Update this path to your dataset directory

clean_dataset(dataset_dir)
