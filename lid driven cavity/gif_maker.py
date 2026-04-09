import imageio.v2 as imageio
import os

images = []
frame_files = sorted(os.listdir("frames"))

for filename in frame_files:
    if filename.endswith(".png"):
        images.append(imageio.imread(os.path.join("frames", filename)))

imageio.mimsave("lid_driven_cavity.gif", images, duration=0.15)