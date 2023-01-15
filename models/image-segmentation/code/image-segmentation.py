"""
Author: Kevin
Github: github.com/loveunk

这是一个image segmentation的例子，使用
Felzenszwalb, Pedro F., and Daniel P. Huttenlocher. 
"Efficient graph-based image segmentation." 
International journal of computer vision 59.2 (2004): 167-181.
中介绍的方法。

关键API：skimage.segmentation.felzenszwalb()
"""

import scipy
import skimage.segmentation
from matplotlib import pyplot as plt

img2 = scipy.misc.imread("manu-2013.jpg", mode="L")
segment_mask1 = skimage.segmentation.felzenszwalb(img2, scale=100)
segment_mask2 = skimage.segmentation.felzenszwalb(img2, scale=1000)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(segment_mask1); ax1.set_xlabel("k=100")
ax2.imshow(segment_mask2); ax2.set_xlabel("k=1000")
fig.suptitle("Felsenszwalb's efficient graph based image segmentation")
plt.tight_layout()
plt.show()