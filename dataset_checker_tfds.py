from absl import app
import tensorflow_datasets as tfds
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from bfm.bfm import BFMModel

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


def DecodeParams(param):
    if len(param) != 62:
        print("wrong length")
        return None, None, None, None
    
    r_dim, t_dim, shape_dim, exp_dim = 9, 3, 40, 10 #62
        
    rotation = param[:r_dim].reshape(3, 3)
    translation = param[r_dim:r_dim+t_dim].reshape(t_dim, 1)
    
    shape = param[r_dim+t_dim:r_dim+t_dim+shape_dim].reshape(shape_dim, 1)
    exp = param[r_dim+t_dim+shape_dim:].reshape(exp_dim, 1)
    
    return rotation, translation, shape, exp

def Reconstruct_vertex_62(bfm, param, imgSize=450):
    R, t, shape, exp = DecodeParams(param)
    
    vertex = (R @ (bfm.u_base + 
                   bfm.w_shp_base @ shape + 
                   bfm.w_exp_base @ exp).reshape(3, -1, order='F') + t)
    vertex[1, :] = imgSize + 1 - vertex[1, :]

    return vertex

def Get_faceBox(vertex, ratio=0.1):
    landmark_2d = (vertex.T)[:, :2].astype(np.float32)
    x, y, w, h = cv2.boundingRect(landmark_2d)
    
    new_w, new_h = int(w * (1+ratio)), int(h * (1+ratio))
    shift_x, shift_y = (new_w - w) // 2, (new_h - h) // 2
    new_x, new_y = int(x - shift_x), int(y - shift_y)
    
    return [new_x, new_y, new_w, new_h]

def Draw_landmarks(image, pts, facebox, outputPath):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    my_dpi = 100
    display_scale = 1 # suggested
    height, width = img.shape[:2]
    figure = plt.figure(figsize=(width / my_dpi, height / my_dpi))
    plt.imshow(img[:, :, ::-1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    if not type(pts) in [tuple, list]:
        pts = [pts]
    
    for i in range(len(pts)):
        alpha = 0.8
        markersize = 1.5
        lw = 0.7 
        color = 'g'
        markeredgecolor = 'green'

        nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

        # close eyes and mouths
        plot_close = lambda i1, i2: plt.plot([pts[i][0, i1], pts[i][0, i2]], [pts[i][1, i1], pts[i][1, i2]],
                                                color=color, lw=lw, alpha=alpha - 0.1)
        plot_close(41, 36)
        plot_close(47, 42)
        plot_close(59, 48)
        plot_close(67, 60)

        for ind in range(len(nums) - 1):
            l, r = nums[ind], nums[ind + 1]
            plt.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

            plt.plot(pts[i][0, l:r], pts[i][1, l:r], marker='o', linestyle='None', markersize=markersize,
                        color=color,
                        markeredgecolor=markeredgecolor, alpha=alpha)
    
    x, y, w, h = facebox        
    xs = [x, x+w, x+w, x, x]
    ys = [y, y, y+h, y+h, y]        
    plt.plot(xs, ys, color='red', lw=lw, alpha=alpha - 0.1)
    
    plt.savefig(outputPath, dpi=my_dpi*display_scale)
    #print('Save landmark result to {}'.format(wfp))
    plt.close()
    

def main(_):
    
    bfm = BFMModel(
        bfm_fp=os.path.join(FILE_DIR, "bfm", "bfm_noneck_v3.pkl"),
        shape_dim=40,
        exp_dim=10
    )
    
    take_percentage = 0.1
    dataset = tfds.load('the300wlp_tfds', 
                        data_dir='the300wlp_tfds',
                        split='train[:{}%]'.format(int(take_percentage*100)))
    
    resultDir = os.path.join(FILE_DIR, "result")
    
    for i, (sample) in enumerate(dataset):
        image = sample['image'].numpy()
        param = sample['param'].numpy()
        
        
        vertex = Reconstruct_vertex_62(bfm, param, image.shape[0])
        facebox = Get_faceBox(vertex)
        
        if not os.path.exists(resultDir):
            os.mkdir(resultDir)
        
        outputPath = os.path.join(resultDir, "{}.jpg".format(i))
        Draw_landmarks(image, vertex, facebox, outputPath)
        
        print("Done drawing {}".format(outputPath))
        
        if not i < 50:
            break

    return

if __name__ == '__main__':
    app.run(main)
