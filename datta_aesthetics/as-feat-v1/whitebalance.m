% Compute difference of image from ideal image with adjusted White Balance
function wb = whitebalance(Image,area_of_frame)
  avg_rgb = mean( reshape(Image, [ area_of_frame,3]) );
  % Find the average gray value and compute the scaling array
  avg_all = mean(avg_rgb);
  wb= max(avg_all, 128)./(avg_rgb+eps);
end