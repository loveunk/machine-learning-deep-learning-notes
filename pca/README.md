# PCA

## Varianes & Covariances 方差 & 协方差
### Variance 方差
<p align="center"><img src="/pca/tex/2c0023d97e0a0cd74c1618cc74f59e4a.svg?invert_in_darkmode&sanitize=true" align=middle width=266.968284pt height=47.60747145pt/></p>
<p align="center"><img src="/pca/tex/9f69ad824f6a6e2c37c1e6776151211d.svg?invert_in_darkmode&sanitize=true" align=middle width=141.76182405pt height=19.726228499999998pt/></p>

### Covariance 协方差
<p align="center"><img src="/pca/tex/87cc7434857420b0abe965cef61647ab.svg?invert_in_darkmode&sanitize=true" align=middle width=414.02777295pt height=17.031940199999998pt/></p>

For 2D data, the Covariance matrix is as follow
<p align="center"><img src="/pca/tex/d0d2ae380c0d5a39d1d9ba0f8452a6a1.svg?invert_in_darkmode&sanitize=true" align=middle width=173.8088847pt height=39.452455349999994pt/></p>

### Rules 方差规则
* <img src="/pca/tex/8807d0472c4891ea38773c7e8fa97e55.svg?invert_in_darkmode&sanitize=true" align=middle width=156.70361354999997pt height=24.65753399999998pt/>
* <img src="/pca/tex/200f2e9e884fc1ac0451588e3164cbbb.svg?invert_in_darkmode&sanitize=true" align=middle width=156.45072464999998pt height=26.76175259999998pt/>

For matrix <img src="/pca/tex/c40a373054f74758866fb2c35baf329d.svg?invert_in_darkmode&sanitize=true" align=middle width=196.09738005pt height=24.65753399999998pt/>
* <img src="/pca/tex/ab091ab2d66cde1f6b86c44b9f80a9d7.svg?invert_in_darkmode&sanitize=true" align=middle width=201.58937039999998pt height=27.6567522pt/>

## Product
### Dot product
#### Algebraic definition 代数定义
<p align="center"><img src="/pca/tex/312d2f414d1bd623f930852764eb9972.svg?invert_in_darkmode&sanitize=true" align=middle width=186.04350929999998pt height=48.18280005pt/></p>

#### Geometric definition 几何定义
<p align="center"><img src="/pca/tex/abe132a79e10c14af7c0790312d3386b.svg?invert_in_darkmode&sanitize=true" align=middle width=148.6376991pt height=18.7598829pt/></p>

### Inner product 内积
定义：对于 <img src="/pca/tex/a0a901384136988a9d6d78e56ddbdbf5.svg?invert_in_darkmode&sanitize=true" align=middle width=58.68325154999999pt height=22.465723500000017pt/>，内积 <img src="/pca/tex/97b1a397fad4dd310d999b59a0255d0d.svg?invert_in_darkmode&sanitize=true" align=middle width=183.67926885pt height=24.65753399999998pt/>，内积具有如下性质：
* Bilinear
  * <img src="/pca/tex/51a7774b54a0c8674fa6daf43423c868.svg?invert_in_darkmode&sanitize=true" align=middle width=203.02496114999997pt height=24.65753399999998pt/>
  * <img src="/pca/tex/12c1b6734072212ed337826ac49018a9.svg?invert_in_darkmode&sanitize=true" align=middle width=203.77074299999998pt height=24.65753399999998pt/>
* Positive definite
  *  <img src="/pca/tex/99c8d938e3892d7c04d582304b66934c.svg?invert_in_darkmode&sanitize=true" align=middle width=210.44457719999997pt height=24.65753399999998pt/>
* Symmetric
  * <img src="/pca/tex/4a48cdf0f489164ef3ba7020880d066a.svg?invert_in_darkmode&sanitize=true" align=middle width=98.1886521pt height=24.65753399999998pt/>

如果定义 <img src="/pca/tex/b7911fe0e9dbfaf20f81aa5ad5a26229.svg?invert_in_darkmode&sanitize=true" align=middle width=100.78174589999998pt height=27.6567522pt/>，当<img src="/pca/tex/6cba520138110bd6f4fe5ebaf7498303.svg?invert_in_darkmode&sanitize=true" align=middle width=42.762416399999985pt height=22.465723500000017pt/>，则其和x，y的点积一致，否则不同。
