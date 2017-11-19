"""
this contains a parser for a training data file.
The training data must contain a list of images.
Let's assume the path to an image is /home/user/..../dataset/images/image.png
Then there MUST be a corresponding label file in /home/user/..../dataset/labels/image.txt

The rule is: replace image file ending with "txt" and replace the name of the folder containing the image with "labels".

The labels must be in the yolo format.
A list of 
label x_center y_center width height
for every object

all coordinates in [0,1]
"""

def parse_image_label_pairs(data_path):
    # creates a list of (image, label) pairs
    # both image and label are filenames
    # the content is loaded dynamically later on
    
    with open(data_path) as train_file:
        image_files = [line.strip() for line in train_file]
        
    # create a list of image, label pairs
    # the list contains tuples of filenames

    image_label_pairs = []

    for image_file in image_files:

        if image_file.strip()=="":
            continue

        if "/" in image_file:
            components = image_file.split("/")
        else:
            components = image_file.split("\\")

        assert len(components)>2, components
        path = "/".join(components[:-2])
        folder = str(components[-2])
        name = str(components[-1])
        name = name.split(".")[0]

        label_file = path+"/labels/"+name+".txt"

        image_label_pairs.append((image_file, label_file))
    
    return image_label_pairs

def parse_labels(label_filename):
    
    objects = []
    with open(label_filename) as file:
        for line in file:
            line = line.strip()
            items = line.split(" ")
            obj = [float(item) for item in items]

            # the coordinates in the label file are
            # x -> right
            # y -> down
            # width for x
            # height for y
            
            # in numpy this is not the same
            # the first dimension is the x dimension
            # but that dimension points downwards
            # and the second, y, points right
            # so we are transposed
            # change the coordinates to align to the numpy layout
            # obj[0] is the label, that stays the same
            # but x and y, as well as w and h are swapped
            obj = [obj[0], obj[2], obj[1], obj[4], obj[3]]

            # we also convert the label to an integer
            obj[0] = int(obj[0])
            
            # now obj[1] is x and obj[3] is size in x dimension
            # now obj[2] is y and obj[4] is size in y dimension
            objects.append(obj)
    return objects