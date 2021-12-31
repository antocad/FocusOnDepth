import numpy as np

class ToMask(object):
    """
        Convert a 3 channel RGB image into a 1 channel segmentation mask
    """
    def __init__(self, palette_dictionnary):
        self.nb_classes = len(palette_dictionnary)

        # sort the dictionary of the classes by the sum of rgb value -> to have always background = 0 
        self.converted_dictionnary = {i: v for i, (k, v) in enumerate(sorted(palette_dictionnary.items(), key=lambda item: sum(item[1])))}

    def __call__(self, pil_image):
        # avoid taking the alpha channel
        image_array = np.array(pil_image)[:, :, :3]

        # get only one channel for the output
        output_array = np.zeros(image_array.shape)[:, :, 0]
        for label in self.converted_dictionnary.keys():
            rgb_color = self.converted_dictionnary[label]
            mask = image_array == rgb_color
            output_array[mask[:, :, 0]] = label
        return output_array