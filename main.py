from human_img_and_skeleton import show_human_and_skeleton
from human_skeleton import human_skeleton
from human_shape import human_shape

img_path = "data/image1.jpg"
display_img = True

# Retun image and mask with 0 and 1
img, mask = human_shape(img_path, display_img) 

# create a skeleton of the human
human_skeleton_img = human_skeleton(mask)

# Display the image
show_human_and_skeleton(img, human_skeleton_img)