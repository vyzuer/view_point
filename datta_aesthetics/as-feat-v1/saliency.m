function smap = saliency(Image, show )
%http://www.klab.caltech.edu/~xhou/projects/spectralResidual/spectralresidual.html
  inImg = im2double(rgb2gray(Image));
  % Spectral Residual
  myFFT = fft2(inImg); 
  myLogAmplitude = log(abs(myFFT));
  myPhase = angle(myFFT);
  mySpectralResidual = myLogAmplitude - imfilter(myLogAmplitude, fspecial('average', 3), 'replicate'); 
  smap = abs(ifft2(exp(mySpectralResidual + 1i*myPhase))).^2;

  % Post processing
  smap = mat2gray(imfilter(smap, fspecial('gaussian', [10, 10], 2.5)));
  if (show)
    imshow(smap);
  end
end