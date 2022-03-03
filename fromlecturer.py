from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

# load 900x900 pixel image, convert the three RGB
# channels to a one channel grey scale
image = Image.open('empire.jpg').convert('L')
print('dimensions ', image.size, 'bit depth: 8bit')

image = np.array(image)  # convert Image object to numpy array

N = 900  # 900x900 image

# create Gaussian point spread function (PSF)
x = np.arange(-float(N/2), float(N/2))
y = x
sigma = 5  # width is 5 pixels
xx, yy = np.meshgrid(x, y, sparse=True)
gauss = (np.exp(-(xx**2+yy**2) / (2 * sigma**2.)))*1

# Gaussian noise that randomises they grey levels
np.random.seed(1)
nwidth = 1e-2
noise = np.reshape(np.array(np.random.normal(
    loc=0, scale=nwidth, size=N**2)), (N, N))
#to remove noise un-comment the next line
#noise=0*noise

# get rid of small numbers (e.g. 1e-40 etc)in the PSF
# if this is not done then get numerical truncation errors and deconv_image is not properly deconvoluted
zero = np.where(gauss < 1e-4)
gauss[zero] = 0

# Fourier transform image and point spread function
fimage = np.fft.fft2(image)
fgauss = np.fft.fft2(gauss)

#convolute in Fourier space and apply inverse transform
#get rid of imaginary parts and turn into real image by applying np.abs
conv_image = np.abs(np.fft.ifft2(fimage*fgauss))


# add noise (for nwidth>1e-4, simple deconvolution fails)
conv_image = conv_image+noise

# do simple deconvolution of  image in Fourier space to restore original image
deconv_image = np.abs(np.fft.ifft2(np.fft.fft2(conv_image)/fgauss))

# construct Wiener filter

#power spectrum of noise
pspec_noise = np.abs(np.fft.fft2(noise))**2

#power spectrum of convoluted image with noise
fconv_image = np.fft.fft2(conv_image)
pconv = np.abs(fconv_image)**2


#Wiener filter

# estimate noise floor from mean of noise power spectrum
# need to impose a noise floor ~6 x mean for Wiener
# filtering to work (note that noise has large variance)
# In practice this would need to be estimated from the power
# spectrum, but here we know the noise floor.
noisefloor = 10*pspec_noise.mean()


# remove noise floor from corrupted image by doing a simple subtraction. As you may get negative values in
# frequency bins in the power spectrum we need to zero all the frequency bins that are below the noise floor
# which is ill defined and leads to problems
# backup power spectrum of noisy, convoluted image for plotting
pconv_withnoise = pconv.copy()
# note that without the copy() method, changes in pconv will lead to the same
# changes in pconv_withnoise as the reference the same object
pconv = pconv-noisefloor  # subtract noise floor from power spectrum
zero = np.where(pconv < 0)  # find negative values in power spectrum
pconv[zero] = 0

# construct Wiener filter with power spectrum of noisy convoluted image where
# noise floor is subtracted and the estimate of the noise floor
wiener = pconv/(pconv+noisefloor)

# do Wiener deconvolution of  image in Fourier space to restore original image
wiener_deconv_image = np.abs(np.fft.ifft2(
    np.fft.fft2(conv_image)*wiener/fgauss))

# convoluted image will be shifted as PSF centered around 450,450
conv_image = np.roll(conv_image, int(N/2), axis=0)
conv_image = np.roll(conv_image, int(N/2), axis=1)


plt.title('Original image')
plt.imshow(image, cmap=plt.get_cmap('gray'))

plt.figure()
plt.title('Gaussian point spread function (PSF) with width sigma')
plt.imshow(gauss, cmap=plt.get_cmap('gray'))

plt.figure()
plt.title('Original image convoluted with PSF')
plt.imshow(conv_image, cmap=plt.get_cmap('gray'))

plt.figure()
plt.title('Simple deconvolution to restore original image')
plt.imshow(deconv_image, cmap=plt.get_cmap('gray'))

plt.figure()
plt.title('Wiener deconvolution to restore original image')
plt.imshow(wiener_deconv_image, cmap=plt.get_cmap('gray'))

# Ideally want to display azimuthally average power spectrum.
# Here we show just one slice of the two dimensional spectrum
plt.figure()
plt.title('Power spectra of convoluted, noisy image and the Gaussian noise')
plt.yscale('log')
# note
freq = np.fft.fftfreq(N)
plt.plot(freq, pconv_withnoise[0, :])
plt.plot(freq, pspec_noise[0, :])
plt.axhline(y=noisefloor, color='r')

plt.figure()
plt.title('Wiener filter')
#plt.yscale('log')
plt.plot(freq, wiener[0, :])
plt.show()
