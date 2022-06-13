import os

image_data_file = 'images/boxes.txt'
path_to_images = 'images/'

def read_image_data():
    """
    Reads from a text file and formats the facial bounding box under each file
    :return: dictionary containing the true bounding boxes of faces in an image
    """
    image_data_dict = {}

    with open(image_data_file) as infile:
        store_filename = False
        prev_file_name = None
        for line in infile:
            dir_file = line.split('/')
            if len(dir_file) > 1:
                full_file_path = path_to_images + line.strip()
                # The line contains a filename
                if os.path.isfile(full_file_path):
                    # Store the next lines in file under this image name
                    store_filename = True
                    prev_file_name = full_file_path
                    image_data_dict[full_file_path] = []
                else:
                    # Ignore data as this file does not exist
                    store_filename = False
                    prev_file_name = None
            else:
                # Line of data
                if store_filename:
                    data = line.split(' ')
                    if len(data) > 4:
                        # Line contains bounding box data
                        # x y w h ...
                        image_data_dict[prev_file_name].append(
                            [int(data[0]), int(data[1]),
                             int(data[2]), int(data[3])])

    return image_data_dict
