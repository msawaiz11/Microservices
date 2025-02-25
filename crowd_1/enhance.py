from PIL import Image, ImageEnhance

# Open the image file
image_path = r"C:\Users\ASDF\Downloads\WhatsApp Image 2024-11-26 at 10.48.58_cae273bf.jpg"
img = Image.open(image_path)

# Auto-enhance steps
# Adjust brightness
brightness_enhancer = ImageEnhance.Brightness(img)
img = brightness_enhancer.enhance(1.2)  # Increase brightness (1.0 = original)

# Adjust contrast
contrast_enhancer = ImageEnhance.Contrast(img)
img = contrast_enhancer.enhance(1.3)  # Increase contrast

# Adjust sharpness
sharpness_enhancer = ImageEnhance.Sharpness(img)
img = sharpness_enhancer.enhance(2.0)  # Sharpen the image

# Adjust color
color_enhancer = ImageEnhance.Color(img)
img = color_enhancer.enhance(1.5)  # Increase color vibrancy

# Save or display the enhanced image
output_path = "enhanced_image.jpg"
img.save(output_path)
img.show()
