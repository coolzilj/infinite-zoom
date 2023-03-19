# Infinite Zoom

## What is this?

This is an extension for [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) which can generate infinite loop videos in minutes.

## How to use it?

### Step 1

To prepare an initial image, you can either generate one using txt2img or find one on the internet. Just ensure that its aspect ratio is 1:1.

### Step 2

Send the image to `img2img/inpaint` and ensure that it has the same width and height. You can refer to my parameters listed below.

![inpaint](./assets/inpaint.png)

From the `Script` dropdown menu, select `Infinite Zoom` and leave it as default if you don't understand what it does.

## Credits

I'm just porting it to AUTOMATIC1111.  
All credit goes to the original creator https://github.com/BalintKomjati/smooth-infinite-zoom.
