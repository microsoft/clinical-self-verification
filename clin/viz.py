import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from IPython.display import display, HTML

def colorize(words: List[str], color_array: np.ndarray[float],
             char_width_max=60, title: str=None, subtitle: str=None):
    '''
    Colorize a list of words based on a color array.
    color_array
        an array of numbers between 0 and 1 of length equal to words
    '''
    cmap = matplotlib.colormaps['viridis_r']
    # cmap = matplotlib.cm.get_cmap('viridis_r')
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    # template = '<span class="barcode"; style="color: {}; background-color: white">{}</span>'
    colored_string = ''
    char_width = 0
    for word, color in zip(words, color_array):
        char_width += len(word)
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
        if char_width >= char_width_max:
            colored_string += '</br>'
            char_width = 0

    if subtitle:
        colored_string = f'<h5>{subtitle}</h5>\n' + colored_string
    if title:
        colored_string = f'<h3>{title}</h3>\n' + colored_string
    return colored_string