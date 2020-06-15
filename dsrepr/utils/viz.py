import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def random_compare(input_images, reconstructed_images, comparisons_per_img=3, n_images=2):
    while comparisons_per_img*n_images>len(input_images):
        n_images -= 1
    if n_images < 1:
        raise ValueError("Invalid parameters for number of images")
    samples = np.random.choice(range(len(input_images)), size=comparisons_per_img*n_images, replace=False)
    idx = 0
    figures = []
    for _ in range(n_images):
        idxs = samples[idx:idx+comparisons_per_img]
        figures.append(compare_reconstruction([input_images[j] for j in idxs],[reconstructed_images[j] for j in idxs]))
        idx += comparisons_per_img
    if len(figures)==1:
        figures = figures[0]
    return figures


def compare_reconstruction(input_images, reconstructed_images, transform=None):
    if not isinstance(input_images, (list, tuple)):
        input_images = [input_images]
    if not isinstance(reconstructed_images, (list, tuple)):
        reconstructed_images = [reconstructed_images]

    assert len(input_images)==len(reconstructed_images)
    assert len(input_images) >= 1

    n_imgs = len(input_images)
    fig, ax = plt.subplots(n_imgs, 2, figsize=(4*2, 4*n_imgs), frameon=False)

    if transform is not None:
        input_images = [transform(img) for img in input_images]
        reconstructed_images = [transform(img) for img in reconstructed_images]
    if n_imgs == 1:
        ax[0].set_title("Input", fontsize=30)
        ax[1].set_title("Reconstruction", fontsize=30)
        ax[0].imshow(np.squeeze(input_images[0]))
        ax[0].set_axis_off()
        ax[1].imshow(np.squeeze(reconstructed_images[0]))
        ax[1].set_axis_off()

    else:
        ax[0, 0].set_title("Input", fontsize=30)
        ax[0, 1].set_title("Reconstr.", fontsize=30)
        for i in range(len(input_images)):
            ax[i, 0].imshow(np.squeeze(input_images[i]))
            ax[i, 0].set_axis_off()
            ax[i, 1].imshow(np.squeeze(reconstructed_images[i]))
            ax[i, 1].set_axis_off()
    return fig
