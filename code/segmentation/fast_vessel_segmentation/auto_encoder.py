import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape
from tensorflow.keras.models import Sequential, Model
from sklearn.mixture import GaussianMixture
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma

np.random.seed(42)

SIZE = 128
LATENT_DIM = 32
img_data = []
img = cv2.imread("ANSYS_UNIGE_19_z0_1.nii.png")
img[img==0] = 49
img_data.append(img_to_array(img))

img_array = np.reshape(img_data, (len(img_data), SIZE, SIZE, 3))
img_array = img_array.astype('float32') / 255.


print(np.min(img_array), np.max(img_array))

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Flatten())
encoded = Dense(LATENT_DIM, activation='relu')(model.layers[-1].output)

# Create the decoder
decoder_input = Input(shape=(LATENT_DIM,))
x = Dense(32 * 32 * 8, activation='relu')(decoder_input)
x = Reshape((32, 32, 8))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
decoder = Model(decoder_input, decoded)

# Combine the encoder and decoder
autoencoder = Model(model.input, decoder(encoded))
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
autoencoder.summary()

autoencoder.fit(img_array, img_array, epochs=200, shuffle=True)


pred = autoencoder.predict(img_array)
plt.imshow(pred[0].reshape(SIZE,SIZE,3), cmap="gray")
plt.show()

# Create a separate encoder model
encoder = Model(model.input, encoded)
# Predict the latent variables
latent_vars = encoder.predict(img_array)
# Flatten the latent variables to 1D for plotting
latent_vars_flat = latent_vars.flatten()



# Plot the histogram
plt.hist(latent_vars_flat, bins=50)
plt.xlabel('Latent Variable Value')
plt.ylabel('Frequency')
plt.show()



# Convert MxNx3 image into Kx3 where K=MxN
img2 = img.reshape((-1,3))  #-1 reshape means, in this case MxN

from sklearn.mixture import GaussianMixture as GMM

#covariance choices, full, tied, diag, spherical
gmm_model = GMM(n_components=2, covariance_type='tied').fit(img2)  #tied works better than full
gmm_labels = gmm_model.predict(img2)

#Put numbers back to original shape so we can reconstruct segmented image
original_shape = img.shape
mask = gmm_labels.reshape(original_shape[0], original_shape[1])
plt.imshow(mask)
plt.show()

vessel = img.copy()
vessel[mask==0] = 0
plt.imshow(vessel)
plt.show()
img_data = []
img_data.append(img_to_array(vessel))

img_array = np.reshape(img_data, (len(img_data), SIZE, SIZE, 3))
img_array = img_array.astype('float32') / 255.

# Predict the latent variables
latent_vars_vessel = encoder.predict(img_array)
# Flatten the latent variables to 1D for plotting
latent_vars_vessel_flat = latent_vars_vessel.flatten()

# Plot the histogram
plt.hist(latent_vars_vessel_flat, bins=50)
plt.xlabel('Latent Variable Value')
plt.ylabel('Frequency')
plt.show()



# bg = img.copy()
# bg[mask>0] = 0
# plt.imshow(bg)
# plt.show()
# img_data = []
# img_data.append(img_to_array(bg))

# img_array = np.reshape(img_data, (len(img_data), SIZE, SIZE, 3))
# img_array = img_array.astype('float32') / 255.

# # Predict the latent variables
# latent_vars_bg = encoder.predict(img_array)
# # Flatten the latent variables to 1D for plotting
# latent_vars_bg_flat = latent_vars_bg.flatten()

# # Plot the histogram
# plt.hist(latent_vars_bg_flat, bins=50)
# plt.xlabel('Latent Variable Value')
# plt.ylabel('Frequency')
# plt.show()


mask[mask > 0] = np.mean(vessel[mask>0])

mask = np.dstack([mask, mask, mask])

img_data = []
img_data.append(img_to_array(mask))

img_array = np.reshape(img_data, (len(img_data), SIZE, SIZE, 3))
img_array = img_array.astype('float32') / 255.

# Predict the latent variables
latent_vars_mask = encoder.predict(img_array)
# Flatten the latent variables to 1D for plotting
latent_vars_mask_flat = latent_vars_mask.flatten()

# Plot the histogram
plt.hist(latent_vars_mask_flat, bins=50)
plt.xlabel('Latent Variable Value')
plt.ylabel('Frequency')
plt.show()


plt.scatter(latent_vars_vessel_flat, [0]*len(latent_vars_vessel_flat), c='blue', label='target')
plt.scatter(latent_vars_flat, [0]*len(latent_vars_flat), c='red', label='original')

plt.xlabel('Latent Variable')
plt.yticks([]) # To remove the y-ticks as they are not needed in a 1D plot
plt.legend()
plt.show()


import seaborn as sns

sns.kdeplot(latent_vars_vessel_flat, shade=True, label='target')
sns.kdeplot(latent_vars_flat, shade=True, label='original')
plt.xlabel('Latent Variable')
plt.ylabel('Density')
plt.legend()
plt.show()



sns.violinplot(data=[latent_vars_vessel_flat, latent_vars_flat])
plt.xticks([0, 1], ['target', 'original'])
plt.ylabel('Latent Variable')
plt.show()


# Enhance the image by denoising and stretching contrast 
from scipy import ndimage as nd
denoise_img = nd.gaussian_filter(img, sigma=2)

denoise_img = np.uint8(denoise_img)
plt.imshow(denoise_img)

from skimage import exposure
p2, p98 = np.percentile(denoise_img, (2, 98))
img2 = exposure.rescale_intensity(denoise_img, in_range=(p2, p98))

img2[img2<=0] = 1


plt.imshow(img2, cmap='gray')
plt.show()
plt.imsave("new.jpg",img2, cmap='gray')


img_data = []
img_data.append(img_to_array(img2))

img_array = np.reshape(img_data, (len(img_data), SIZE, SIZE, 3))
img_array = img_array.astype('float32') / 255.

# Predict the latent variables
latent_vars_new = encoder.predict(img_array)
# Flatten the latent variables to 1D for plotting
latent_vars_new_flat = latent_vars_new.flatten()


plt.scatter(latent_vars_vessel_flat, [0]*len(latent_vars_vessel_flat), c='blue', label='target')
plt.scatter(latent_vars_mask_flat, [0]*len(latent_vars_mask_flat), c='black', label='mask')
plt.scatter(latent_vars_flat, [0]*len(latent_vars_flat), c='red', label='original')
plt.scatter(latent_vars_new_flat, [0]*len(latent_vars_flat), c='green', label='new')

plt.xlabel('Latent Variable')
plt.yticks([]) # To remove the y-ticks as they are not needed in a 1D plot
plt.legend()
plt.show()


import seaborn as sns

sns.kdeplot(latent_vars_vessel_flat, shade=True, label='target')
sns.kdeplot(latent_vars_mask_flat, shade=True, label='mask')
sns.kdeplot(latent_vars_flat, shade=True, label='original')
sns.kdeplot(latent_vars_new_flat, shade=True, label='new')
plt.xlabel('Latent Variable')
plt.ylabel('Density')
plt.legend()
plt.show()



sns.violinplot(data=[latent_vars_vessel_flat, latent_vars_mask_flat, latent_vars_flat, latent_vars_new_flat])
plt.xticks([0, 1, 2, 4], ['target', 'mask', 'original', 'new'])
plt.ylabel('Latent Variable')
plt.show()


