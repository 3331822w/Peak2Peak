# Peak2Peak
Revealing the denoising principle of zero-shot N2N-based algorithm from 1D spectrum to 2D image 
Denoising is a necessary step in the image analysis to extract weak signals, especially those hardly identified by the naked eye. Unlike the traditional denoising algorithms relying on a clean image as the reference, Noise2Noise (N2N) was able to denoise the noisy image, providing sufficiently noisy images with the same subject but randomly distributed noise. Further, by introducing data augmentation to create a big dataset and regularization to prevent model overfitting, zero-shot N2N-based denoising was proposed in which only a single noisy image was needed. Although various N2N-based denoising algorithms have been developed with high performance, their complicated black box operation prevented the lightweight. Therefore, to reveal the working function of the zero-shot N2N-based algorithm, we proposed a lightweight Peak2Peak algorithm (P2P), and qualitatively and quantitatively analyzed its denoising behavior on the 1D spectrum and 2D image. We found that the high-performance denoising originated from the trade-off balance between loss function and regularization in the denoising module, where regularization is the switch of denoising. Meanwhile, the signal extraction is mainly from the self-supervised characteristic learning in the data augmentation module. Further, the lightweight P2P improved the denoising speed by at least ten times but with little performance loss, compared with that of the current N2N-based algorithms. In general, the visualization of P2P provides a reference for revealing the working function of zero-shot N2N-based algorithms, which would pave the way for the application of these algorithms towards real-time (in-situ, in-vivo, and operando) research improving both temporal and spatial resolutions.
