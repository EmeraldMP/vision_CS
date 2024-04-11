import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

import matplotlib.image as mpimg
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


def find_critical_points(skel):
    # Kernel for detecting endpoints: if the sum is 2 (center pixel + 1 neighbor), it's an endpoint
    endpoint_kernel = np.array([[1, 1, 1],
                                [1, 10, 1],
                                [1, 1, 1]], dtype=np.uint8)

    # Kernel for detecting junctions: if the sum is greater than 3, it's a junction
    junction_kernel = np.array([[1, 1, 1],
                                [1, 10, 1],
                                [1, 1, 1]], dtype=np.uint8)

    # Finding the endpoints and junctions
    endpoints = (convolve(skel // 255, endpoint_kernel) == 11) * 1
    junctions = (convolve(skel // 255, junction_kernel) > 11) * 1

    return endpoints, junctions

def mark_critical_points(skel, endpoints, junctions):
    # Create an RGB version of the gray scale skeleton image
    skel_color = np.stack((skel,)*3, axis=-1)

    # Marking the endpoints in red and junctions in blue
    skel_color[endpoints == 1] = [255, 0, 0]
    skel_color[junctions == 1] = [0, 0, 255]

    return skel_color

def find_intersections(skel):
    # Define a kernel that will help identify intersection (branching) points
    # Intersection points will have 3 or more neighboring pixels
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # Apply convolution
    neighbor_count = convolve(skel // 255, kernel)
    # Intersection points are where the count in the result is 13 or more (center + 3 or more neighbors)
    intersections = (neighbor_count >= 13) * 1
    return intersections


#def plot_point_of_interest(skeleton):
def plot_point_of_interest(skel_thresh):
    # Read the skeleton image from file
    
    # Threshold the image to make sure it's binary
    # _, skel_thresh = cv2.threshold(skeleton, 127, 255, cv2.THRESH_BINARY)
    
    # Find the critical points on the skeleton
    endpoints, junctions = find_critical_points(skel_thresh)
    
    # Create an image with marked critical points
    marked_img = mark_critical_points(skel_thresh, endpoints, junctions)
    
    # Find coordinates of critical points for plotting
    y_endpoints, x_endpoints = np.where(endpoints == 1)
    y_junctions, x_junctions = np.where(junctions == 1)
    
    intersections = find_intersections(skel_thresh)
    
    # Create an image with marked critical points
    marked_img = mark_critical_points(skel_thresh, endpoints, junctions)
    
    # Find coordinates of intersection points for plotting
    y_intersections, x_intersections = np.where(intersections == 1)
    
    # Use matplotlib to display the image with larger markers
    plt.imshow(marked_img)
    plt.scatter(x_junctions, y_junctions, c='b', s=4)  # Blue junctions
    plt.scatter(x_endpoints, y_endpoints, c='r', s=40)  # Red endpoints
    plt.scatter(x_intersections, y_intersections, c='g', s=40)  # Green intersections
    plt.title('Skeleton with Critical and Intersection Points')
    plt.axis('off')  # Hide axis ticks and labels
    plt.show()

def resize_image(original_image, display_=True):    
    # Get the dimensions of the original image
    height, width = original_image.shape[:2]
    
    # Calculate the new dimensions by dividing the original dimensions by 16
    new_height, new_width = height // 4, width // 4
    
    # Resize the image to the new dimensions using linear interpolation
    resized_image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    print(resized_image.shape)  # This will output (height, width)
    cv2.imwrite('Images/resized_image.jpg', resized_image)
    # Plot the original and resized images

    if display_:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        plt.title('Resized Image')
        plt.axis('off')
        
        plt.show()

    return resized_image


def zhang_suen_thinning(image, display_,display_only_last):
    # Function to perform one iteration of Thinning with the given iter flag
    def thinning_iteration(img, iter):
        marker = np.ones(img.shape, dtype=np.uint8)
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                '''
                p9 p8 p7
                p2 -  p6
                p3 p4 p5
                '''
                p2 = img[i-1, j]# & marker[i-1,j]
                p3 = img[i-1, j+1]# & marker[i-1,j-1]
                p4 = img[i, j+1]# & marker[i,j+1]
                p5 = img[i+1, j+1]# & marker[i+1,j+1]
                p6 = img[i+1, j]# & marker[i+1,j]
                p7 = img[i+1, j-1]# & marker[i+1,j-1]
                p8 = img[i, j-1]# & marker[i,j-1]
                p9 = img[i-1, j-1]# & marker[i-1,j-1]

                if p2 == 0: 
                    p2 = False
                else:
                    p2 = True
                if p3 == 0: 
                    p3 = False
                else:
                    p3 = True
                if p4 == 0: 
                    p4 = False
                else:
                    p4 = True
                if p5 == 0: 
                    p5 = False
                else:
                    p5 = True
                if p6 == 0: 
                    p6 = False
                else:
                    p6 = True
                if p7 == 0: 
                    p7 = False
                else:
                    p7 = True
                if p8 == 0: 
                    p8 = False
                else:
                    p8 = True
                if p9 == 0: 
                    p9 = False
                else:
                    p9 = True
                
                A  = (p2 == 0 and p3 == 1) + (p3 == 0 and p4 == 1) + \
                     (p4 == 0 and p5 == 1) + (p5 == 0 and p6 == 1) + \
                     (p6 == 0 and p7 == 1) + (p7 == 0 and p8 == 1) + \
                     (p8 == 0 and p9 == 1) + (p9 == 0 and p2 == 1)
                
                B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                m1 = (p2 * p4 * p6) if iter == 0 else (p2 * p4 * p8)
                m2 = (p4 * p6 * p8) if iter == 0 else (p2 * p6 * p8)
                
                e0 = (p2 and p5)
                e1 = (p2 and p6)
                e2 = (p2 and p7)
                e3 = (p6 and p9)
                e4 = (p6 and p3)

                n0 = (p4 and p7)
                n1 = (p4 and p8)
                n2 = (p4 and p9)
                n3 = (p8 and p3)
                n4 = (p8 and p5)

                d0 = (p9 and p5)
                d1 = (p3 and p7)

                n = n0 or n1 or n2 or n3 or n4
                e = e0 or e1 or e2 or e3 or e4
                d = d0 or d1

                # Alguno de estos es True
                horizontal0 = e0 and not e1 and not e2 and not e3 and not e4 and not n and not d
                horizontal1 = not e0 and e1 and not e2 and not e3 and not e4 and not n and not d
                horizontal2 = not e0 and not e1 and e2 and not e3 and not e4 and not n and not d
                horizontal3 = not e0 and not e1 and not e2 and e3 and not e4 and not n and not d
                horizontal4 = not e0 and not e1 and not e2 and not e3 and e4 and not n and not d

                vertical0   = n0 and not n1 and not n2 and not n3 and not n4 and not e and not d
                vertical1   = not n0 and n1 and not n2 and not n3 and not n4 and not e and not d
                vertical2   = not n0 and not n1 and n2 and not n3 and not n4 and not e and not d
                vertical3   = not n0 and not n1 and not n2 and n3 and not n4 and not e and not d
                vertical4   = not n0 and not n1 and not n2 and not n3 and n4 and not e and not d

                diagonal0   = not n and not e and d0 and not d1
                diagonal1   = not n and not e and not d0 and d1

                # Alguno de estos es True
                conecta_horizontal = horizontal0 or horizontal1 or horizontal2 or horizontal3 or horizontal4
                conecta_vertical   = vertical0   or vertical1   or vertical2   or vertical3   or vertical4
                conecta_diagonal   = diagonal0   or diagonal1
                
                concecta = conecta_horizontal or conecta_vertical or conecta_diagonal

                if i == 107 and j == 60:
                    print('p2=', p2, 'p3=', p3, 'p4=', p4, 'p5=', p5, 'p6=', p6, 'p7=', p7, 'p8=', p8, 'p9=', p9)
                    #print('A=', A)
                    #print('B=',B)
                    print('n0',n0)
                    #print('conecta_horizontal', conecta_horizontal, conecta_vertical, conecta_diagonal)
                    print('horizontal0 ', horizontal0)#,horizontal1,horizontal2,horizontal3,horizontal4)
                    print('e1 :', e1, not e1)


                # No está entrando al loop -> conecta = True
                if A == 1 and (B >= 2 and B <= 6) and not concecta:
                    marker[i,j] = 0

        return img & marker

    
    # Make sure the image is binary
    image = image // 255
    prev = np.zeros(image.shape, np.uint8)
    diff = None
    iter = 0
    while True:
        iter += 1
        # Two iterations: one for sub-iteration 0, another for sub-iteration 1
        image = thinning_iteration(image, 0)
        image = thinning_iteration(image, 1)
        # Check if there is no difference between the current and the previous iteration
        diff = np.sum(abs(image - prev))
        if diff == 0:
            break
        prev = image.copy()
        if iter % 1 == 0 and display_:
            plt.figure(figsize=(10, 10))
            plt.imshow(image * 255, cmap='gray')
            plt.axis('off')
            plt.title('Skeleton of the Human Shape')
            plt.show()

    if not display_only_last:
        plt.figure(figsize=(10, 10))
        plt.imshow(image * 255, cmap='gray')
        plt.axis('off')
        plt.title('Skeleton of the Human Shape')
        plt.show()

    return image * 255  # return image to original scale

# We'll first recompute the minimum spanning tree.
def compute_steiner_tree(points):
    # Convert points list to a numpy array if it's not already
    points_array = np.array(points)
    # Create the distance matrix
    dist_matrix = squareform(pdist(points_array))
    # Create the graph
    graph = csr_matrix(dist_matrix)
    # Compute the minimum spanning tree (MST) of the graph
    mst = minimum_spanning_tree(graph)
    return mst.toarray(), points_array


# This function will plot the Steiner tree with a -90° rotation to correct the orientation.
def plot_steiner_tree_rotated(mst, points_array):
    # Apply a -90° rotation around the origin (0,0)
    theta = np.radians(90)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    rotated_points = points_array @ R

    rotated_points = rotated_points * 4

    plt.figure(figsize=(10, 10))
    plt.scatter(rotated_points[:, 0], rotated_points[:, 1], c='yellow')  # Points in yellow

    # Draw the edges of the MST
    for start_point in range(len(rotated_points)):
        for end_point in range(len(rotated_points)):
            if mst[start_point, end_point] > 0:
                plt.plot([rotated_points[start_point][0], rotated_points[end_point][0]],
                         [rotated_points[start_point][1], rotated_points[end_point][1]], 'ro-')

    plt.axis('off')  # Hide axes
    plt.show()


def plot_steiner_tree_on_image(mst, points, img):
    # Apply a -90° rotation around the origin (0,0)
    theta = np.radians(-90)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    # Rotate the points
    rotated_points = points @ R
    #rotated_points *= 4
    # Invert the x-axis by multiplying by -1
    rotated_points[:, 0] *= -1
    
    # Find the bounding box of the rotated points
    x_min, y_min = rotated_points.min(axis=0)
    x_max, y_max = rotated_points.max(axis=0)
    
    # Determine the translation needed to center the skeleton on the image
    x_trans = (img.shape[1] - (x_max + x_min)) / 2
    y_trans = (img.shape[0] - (y_max + y_min)) / 2
    translation = np.array([x_trans, y_trans])

    # Apply the translation to the points
    translated_points = rotated_points# + translation
    translated_points = cv2.resize(translated_points, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
    
    # Now, plot the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    
    # Overlay the Steiner tree skeleton
    plt.scatter(translated_points[:, 0], translated_points[:, 1], c='blue', s=10)  # Points in yellow

    # Draw the edges of the MST
    for start_point in range(len(translated_points)):
        for end_point in range(len(translated_points)):
            if mst[start_point, end_point] > 0:
                plt.plot([translated_points[start_point][0], translated_points[end_point][0]],
                         [translated_points[start_point][1], translated_points[end_point][1]], 'r-')

    plt.axis('off')  # Hide axes
    plt.show()


def long_human_skeleton(original_img, binary_image, display_ = False, display_only_last=True):

    # 1
    
    # Apply Euclidean Distance Transform
    edt_image = distance_transform_edt(binary_image)
    
    # Skeletonize
    skeleton = skeletonize(binary_image>0)
    
    # Convert skeleton to a format that can be saved as an image
    skeleton_image = np.uint8(skeleton) * 255
    
    # Display the image
    if display_:
        plt.imshow(skeleton_image, cmap='gray')
        plt.axis('off')  # Turn off the axis
        plt.title('Skeleton of the Human Shape')
        plt.show()


    # 2
    if not display_only_last:
        plot_point_of_interest(skeleton_image)

    # 3

    #resized_image = resize_image(original_image, display_)
    #resized_image = resize_image(binary_image, display_)
    resized_image = binary_image
    # 4

    original_image = 255 - resized_image
    
    # Perform skeletonization
    skeleton = zhang_suen_thinning(binary_image, display_, display_only_last)
    
    # Show the skeleton
    if display_:
        plt.figure(figsize=(10, 10))
        plt.imshow(skeleton, cmap='gray')
        plt.axis('off')
        plt.title('Skeleton of the Human Shape')
        plt.show()

    # 5
    
    points = []
    for i in range(len(skeleton)):
        for j in range(len(skeleton[0])):
            if skeleton[i][j] > 0:
                points.append((i,j))
  
    mst, points_array = compute_steiner_tree(points)
    #print(nodes_degree_one)

    
    # Now, plot the rotated MST
    if display_:
        plot_steiner_tree_rotated(mst, points_array)

    # 6
    
    #image_path = 'Images/shape humana.png'
    #original_image = cv2.imread(image_path, 0)
    original_image = original_img

    
    plot_steiner_tree_on_image(mst, points, original_image)


def human_skeleton(original_img, binary_image, display_ = False, display_only_last=True):

    # 1
  
    original_image = 255 - binary_image
    
    # Perform skeletonization
    skeleton = zhang_suen_thinning(binary_image, display_, display_only_last)
    
    # Show the skeleton
    if display_:
        plt.figure(figsize=(10, 10))
        plt.imshow(skeleton, cmap='gray')
        plt.axis('off')
        plt.title('Skeleton of the Human Shape')
        plt.show()

    # 5
    
    points = []
    for i in range(len(skeleton)):
        for j in range(len(skeleton[0])):
            if skeleton[i][j] > 0:
                points.append((i,j))
  
    mst, points_array = compute_steiner_tree(points)

    # Now, plot the rotated MST
    if display_:
        plot_steiner_tree_rotated(mst, points_array)

    # 6

    original_image = original_img

    plot_steiner_tree_on_image(mst, points, original_image)


if __name__ == "__main__":
    human_skeleton('Images/shape humana.png', True, False)
