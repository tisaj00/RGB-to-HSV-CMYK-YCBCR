import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

#učitavanje slike
img=mpimg.imread('C:/Users/Josip/Desktop/Diplomski moje/OSIRV PROJEKT/slike/zec.png',0)

rgb_scale = 255
cmyk_scale = 100

#funkcija za prebacivanje u CMYK format
def rgb_to_cmyk(rgb, percent_gray=100):


    cmy = 1 - rgb / 255.0
    k = np.min(cmy, axis=2) * (percent_gray / 100.0)
    k[np.where(np.sum(rgb,axis=2)==0)] = 1.0  # anywhere there is no color, set the k chanel to max
    k_mat = np.stack([k,k,k], axis=2)

    with np.errstate(divide='ignore', invalid='ignore'):
        cmy = (cmy - k_mat) / (1.0 - k_mat)
        cmy[~np.isfinite(cmy)] = 0.0

    return np.dstack((cmy, k))

#funkcija za prebacivanje u HSV format
def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv


#funkcija za prebacivanje u YCBCR format
def rgb_to_ycbcr(im):
    cbcr = np.empty_like(im)
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]
    # Y
    cbcr[:,:,0] = .299 * r + .587 * g + .114 * b
    # Cb
    cbcr[:,:,1] = 128 - .169 * r - .331 * g + .5 * b
    # Cr
    cbcr[:,:,2] = 128 + .5 * r - .419 * g - .081 * b
    return np.uint8(cbcr)


#prikaz RGB slike
plt.title('RGB image') 
plt.imshow(img)
plt.show()

#prikaze YCBCR slike
plt.title('YCBCR image')
img1=rgb_to_ycbcr(img)
plt.imshow(img1)
plt.savefig('C:/Users/Josip/Desktop/Diplomski moje/OSIRV PROJEKT/slike/YCBCR_Zec.png')
plt.show()

#prikaze HSV slike
plt.title('HSV image')
img2=rgb_to_hsv(img)
plt.imshow(img2)
plt.savefig('C:/Users/Josip/Desktop/Diplomski moje/OSIRV PROJEKT/slike/HSV_Zec.png')
plt.show()

#prikaze CMYK slike
plt.title('CMYK image')
img2=rgb_to_cmyk(img)
plt.imshow(img2)
plt.savefig('C:/Users/Josip/Desktop/Diplomski moje/OSIRV PROJEKT/slike/CMYK_Zec.png')
plt.show()