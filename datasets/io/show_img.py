import ipywidgets

from IPython.display import clear_output, display

def show_img(img_ix, img_list, read_fn=None, display_fn=None):
    """function to display image with previous/next buttons
    img_ix: int initial index of the image to be displayed
    img_list: np.array list of image data
    read_fn: function used to read image files
    display_fn: function used to display image files
    """

    if img_ix < 0:
        img_ix = len(img_list) + img_ix
    elif img_ix >= len(img_list):
        img_ix = img_ix - len(img_list)
        
    print(img_ix)
    x = img_list[img_ix]

    if read_fn is not None:
        x = read_fn(x)
    
    def on_button_next(b):
        clear_output()
        show_img(img_ix + 1, img_list, read_fn)

    def on_button_prev(b):
        clear_output()
        show_img(img_ix - 1, img_list, read_fn)

    button_next = ipywidgets.Button(description="Next")
    button_prev = ipywidgets.Button(description="Prev")
    display(ipywidgets.HBox([button_prev, button_next]))
    button_next.on_click(on_button_next)
    button_prev.on_click(on_button_prev)
    
    return display_fn(x)
