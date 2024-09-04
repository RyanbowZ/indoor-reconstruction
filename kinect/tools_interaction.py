
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os



btn_pos = np.zeros((1,2))
def callback(event):
    global btn_pos
    if event.inaxes is not None:
        # print (event.xdata, event.ydata)
        btn_pos[0] = [event.xdata, event.ydata]
        plt.close()
    else:
        print( 'Clicked ouside axes bounds but inside plot window')


def get_color(img_path, annotations_path,out_dir):
    # interactively select color in cv2 GUI
    # get color by click, and save it to out_dir
    global btn_pos
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    anno = cv2.imread(annotations_path)
    anno = cv2.cvtColor(anno, cv2.COLOR_BGR2RGB)
    fig, ax = plt.plot()
    ax.imshow(image)
    fig.canvas.callbacks.connect('button_press_event', callback)
    plt.show()  
    color = anno[btn_pos[0][1].astype(int), btn_pos[0][0].astype(int)]
    os.makedirs(out_dir, exist_ok=True)
    np.save(out_dir/"color.npy", color)
