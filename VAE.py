"""
Author: Max Martinez Ruts
Date: January 2019

Description:

"""



from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import pygame

"""
Reparameterization trick by sampling fr an isotropic unit Gaussian.
Arguments: (tensor): mean and log of variance of Q(z|X)
Returns: z (tensor): sampled latent vector
"""
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # Random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Generate data
display = pygame.display.set_mode((350, 350))

# Convert cartesian coordinates to screen coordinates
def cartesian_to_screen(car_pos):
    factor = 1
    screen_pos = np.array([center[0] * factor + car_pos[0], center[1] * factor - car_pos[1]]) / factor
    screen_pos = screen_pos.astype(int)
    return screen_pos

# Convert sceen coordinates to cartesian coordinates
def screen_to_cartesian(screen_pos):
    factor = 1
    car_pos = np.array([screen_pos[0] - center[0], center[1] - screen_pos[1]]) * factor
    car_pos = car_pos.astype(float)
    return car_pos

# Draw an image in pygame given a matrix of pixel brightness
def draw(img):
    pygame.event.get()
    screen.fill((0, 0, 0))
    # For each pixel
    for i in range(0, 28):
        for j in range(0, 28):

            # Pixel brightness is value of cell in matrix
            color = int(img[j,i]*255)
            pixel = pygame.Rect(i*width/28,j*width/28, width/28, width/28)

            # Draw Pixel
            pygame.draw.rect(screen, (color,color,color), pixel)

    # Draw in screen
    pygame.display.flip()

# Screen parameters
width = 28
height = 28
center = np.array([width/2, height/2])
screen = pygame.display.set_mode((width, height))

# Colors
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
white = (255, 255, 255)
yellow = (255,255, 0)

# Number of images in dataset
n_samples = 25000

# Generate random coordinates following a normal 2D distribution
xs = np.random.randn(n_samples)*3
ys = np.random.randn(n_samples)*3

# Initialize dataset
data = np.zeros((n_samples,28,28))

# For each image
for i in range(n_samples):
    pygame.event.get()

    screen.fill((0, 0, 0))

    # Draw circle in screen
    pygame.draw.circle(screen, white, cartesian_to_screen(np.array([xs[i],ys[i]])), 3)
    pygame.display.flip()

    display.blit(screen, (0, 0))
    pygame.display.update()

    # Convert the window in black color(2D) into a matrix
    screen_px = pygame.surfarray.array2d(display)
    screen_px = screen_px/ np.max(screen_px)
    screen_px = np.flip(np.rot90(np.rot90(np.rot90(screen_px))),axis=1)

    # Save matrix in dataset
    data[i] = screen_px

# Divide dataset in train and test
x_train = data[:int(9/10*n_samples),:,:].astype('float32')
x_test = data[int(9/10*n_samples):,:,:].astype('float32')
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Image parameters
image_size = 28
original_dim = image_size * image_size

# Network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = 2
epochs = 1

# Create input layer
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)

# Generate layers containing the mean and variance of x
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# Use reparameterization trick to push the sampling out as input
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# Instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# Build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# Instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# Instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

# Build loss function
reconstruction_loss = mse(inputs, outputs)
reconstruction_loss *= original_dim

# Personalized function (modified parameters to improve efficiency)
kl_loss = 1 + z_log_var - K.square(z_mean) -10*K.exp(z_log_var*10)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

vae.compile(optimizer='adam')
vae.summary()

# Train the autoencoder
vae.fit(x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

# Encode input and then fit the encoded values into the decoder to get the decoded values
x_encoded = encoder.predict(x_test,batch_size=batch_size)
x_decoded = decoder.predict(x_encoded[2])

x_mean = np.mean(x_encoded[2], axis=0)
x_stds = np.std(x_encoded[2], axis=0)
x_cov = np.cov((x_encoded[2] - x_mean).T)
e, v = np.linalg.eig(x_cov)
e_list = e.tolist()
e_list.sort(reverse=True)


xs_test = xs[int(9 / 10 * n_samples):]
ys_test = ys[int(9 / 10 * n_samples):]
# Create a scatter plot of the 2D coordinates of all points.
plt.figure(figsize=(6, 6))
plt.scatter(xs_test, ys_test, c=x_encoded[2][:, 0])
plt.colorbar()
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(xs_test, ys_test, c=x_encoded[2][:, 1])
plt.colorbar()
plt.show()

# Modify screen parameters (larger screen)
width = 600
height = 600
center = np.array([width / 2, height / 2])
screen = pygame.display.set_mode((width, height))
k = 0





# Visualize the effect of the latent values when they are fed to the decoder. The cartesian coordinates of the mouse position represent the two latent values
while True:
    for event in pygame.event.get():
        # When click event
        if event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            cartesian_pos = screen_to_cartesian(mouse_pos)
            pygame.event.get()
            screen.fill((0, 0, 0))

            # Draw circle in screen
            pygame.draw.circle(screen, white, cartesian_to_screen(np.array([cartesian_pos[0], cartesian_pos[1]])), int(3*21.4285714))
            pygame.display.flip()

            display.blit(screen, (0, 0))
            pygame.display.update()

            # Convert the window in black color(2D) into a matrix
            screen_px = pygame.surfarray.array2d(display)
            screen_px = screen_px / np.max(screen_px)
            screen_px = np.flip(np.rot90(np.rot90(np.rot90(screen_px))), axis=1)
            # draw(screen_px)

            #
            #
            screen_px.reshape((1, 28*28))
            x_encoded = encoder.predict(screen_px, batch_size=batch_size)
            x_decoded = decoder.predict(x_encoded[2])

            digit = x_decoded[0].reshape(28, 28)
            draw(digit)

rand_vecs = np.zeros(latent_dim)

# Visualize the effect of the latent values when they are fed to the decoder. The cartesian coordinates of the mouse position represent the two latent values
while True:
    for event in pygame.event.get():
        # When click event
        if event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos

            cartesian_pos = screen_to_cartesian(mouse_pos)
            rand_vecs = cartesian_pos / 300
            rand_vecs = np.array(rand_vecs)
            rand_vecs = list(rand_vecs)
            for i in range(2, latent_dim):
                rand_vecs.append(0)
            rand_vecs = np.array(rand_vecs)

    # z_sample = rand_vecs
    # z_sample = x_mean + np.dot(v, (rand_vecs * e).T).T

    x_decoded = decoder.predict(np.array([rand_vecs]))
    digit = x_decoded[0].reshape(28, 28)
    draw(digit)
