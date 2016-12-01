% Function to compute Colorfulness in a given image
function [counter_colorfulness] = Colorfulness(Image)
  counter_colorfulness = 0;
  HSV_Im = rgb2hsv(Image);
  Count = imhist(HSV_Im(:,1),360);
  for i=1:360
    if Count(i)> 0
      counter_colorfulness = counter_colorfulness+1;
     end
   end
return
