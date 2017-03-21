caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import surgery, score
from scipy.misc import imresize, imsave, toimage
import time


# Run in GPU
caffe.set_device(0)
caffe.set_mode_gpu()

#Compute heatmaps from images in txt
val = np.loadtxt('../../datasets/COCO-Text/val-onlyLegibleText.txt', dtype=str)

# load net
net = caffe.Net('voc-fcn8s-atonce/deploy.prototxt', '../models/fcn-COCOText/train_iter_104000.caffemodel', caffe.TEST)


print 'Computing heatmaps ...'

count = 0
start = time.time()

for idx in val:

    count = count + 1
    if count % 100 == 0:
        print count

    # load image
    im = Image.open('../../datasets/COCO-Text/val-onlyLegibleText/' + idx + '.jpg')
    im_o = im
    im = im.resize((512, 512), Image.ANTIALIAS)

    # Turn grayscale images to 3 channels
    if (im.size.__len__() == 2):
        im_gray = im
        im = Image.new("RGB", im_gray.size)
        im.paste(im_gray)

    #switch to BGR and substract mean
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W)
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

    # run net and take scores
    net.forward()

    # Compute SoftMax HeatMap
    hmap_0 = net.blobs['score_conv'].data[0][0, :, :]   #Text score
    hmap_1 = net.blobs['score_conv'].data[0][1, :, :]   #Backgroung score
    hmap_0 = np.exp(hmap_0)
    hmap_1 = np.exp(hmap_1)
    hmap_softmax = hmap_1 / (hmap_0 + hmap_1)

    #Save CSV heatmap
    # pixels = np.asarray(hmap_softmax)
    # np.savetxt('/home/imatge/caffe-master/data/coco-text/csv_heatmaps/voc-fcn8s-atonce-104000it/' + idx + '.csv', pixels, delimiter=",")

    #Save PNG softmax heatmap
    hmap_softmax_2save = (255.0 * hmap_softmax).astype(np.uint8)
    hmap_softmax_2save = Image.fromarray(hmap_softmax_2save)
    hmap_softmax_2save = hmap_softmax_2save.resize((im_o.size[0], im_o.size[1]), Image.ANTIALIAS)
    hmap_softmax_2save.save('../../datasets/COCO-Text/val-onlyLegibleText-heatmaps/' + idx + '.png')


    # Save color softmax heatmap
    # fig = plt.figure(frameon=False)
    # fig.set_size_inches(im.size[0]/8,im.size[1]/8)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # ax.imshow(hmap_softmax, aspect='auto', cmap="jet")
    # fig.savefig('../data/supermarket/heatmaps/' + idx)
    # plt.close(fig)


    print 'Heatmap saved for image: ' +idx

end = time.time()
print 'Total time elapsed in heatmap computations'
print(end - start)
print 'Time per image'
print(end - start)/val.__len__()