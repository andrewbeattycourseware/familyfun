from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')
image1_filename = "frame1.png"
image2_filename = "frame2.png"

#image = Image.open(image1filename).convert('L')
image1 = Image.open(image1_filename)
image2 = Image.open(image2_filename)


image1_width, image1_height = image1.size
print('dimensions ', image1_width, ',',  image1_height)
#print('sanity check', image2.size )

# convert Image object to numpy array and convert to float
#image1_array = np.array(image1) / 1.0
#image2_array = np.array(image2) /1.0
# this should be a better way than dividing by 1.0 as recommended by question
# reference https://www.geeksforgeeks.org/using-numpy-to-convert-array-elements-to-float-type/
image1_array = np.array(image1).astype(np.float)
image2_array = np.array(image2).astype(np.float)


#print (image1_array)
#print (image1_array[0].shape)
numberpixels = np.empty(image1_height)

for i in range(0,image1_height):
    
    #correlations =  np.correlate(image1_array[i], image2_array[i], mode='full')
    #correlations =  np.correlate(image2_array[i], image1_array[i], mode='full')[image1_width:]
    correlations =  np.correlate(image2_array[i], image1_array[i], mode='full')
    
    
    #print (correlations.shape)
    #https: // www.datasciencelearner.com/find-max-and-min-value-of-numpy-array/
    max = np.max(correlations)
    conditon = (correlations == max)
    #print  (np.where(conditon)[0])
    numberpixels[i] = np.where(conditon)[0] - (image1_width - 1)
    #print( np.correlate(image1_array[i].flatten(), image2_array[i].flatten()))

#print (correlations)
xpoints = np.array(range(0, image1_height))
plt.plot(xpoints, numberpixels)
plt.savefig("gragh_best.png")
plt.show()

