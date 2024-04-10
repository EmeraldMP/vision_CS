from human_skeleton import human_skeleton
from human_shape import human_shape

img_path = "Images/image1.jpg"
display_img = False

# Retun image and mask with 0 and 1
img, mask = human_shape(img_path, display_img) 

print(img.shape, mask.shape)

# create a skeleton of the human
human_skeleton_img = human_skeleton(img, mask, True, False)

# Display the image
show_human_and_skeleton(img, human_skeleton_img)