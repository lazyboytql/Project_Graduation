import cv2
import os

def convert_to_gray(input_path_color, output_path_gray, num_images):
    for i in range(1, num_images + 1):
        input_image_path = f'{input_path_color}/color_image_{i}.jpg'
        
        # Check if the image exists
        if os.path.exists(input_image_path):
            # Read the color image from the path
            color_image = cv2.imread(input_image_path)

            # Convert the color image to grayscale
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # Save the grayscale image
            cv2.imwrite(f'{output_path_gray}/gray_image_{i}.jpg', gray_image)
        else:
            print(f'Image {i} not found.')

# Replace the paths below with the directory containing color images and the directory to save grayscale images
input_path_color = r'D:\Project_Graduation\image\origin'
output_path_gray = r'D:\Project_Graduation\image\gray'
num_images = 77  # Number of color images

# Create a directory to save grayscale images if it does not exist
os.makedirs(output_path_gray, exist_ok=True)

# Call the function to convert color images to grayscale
convert_to_gray(input_path_color, output_path_gray, num_images)
