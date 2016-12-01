% Computes depth of field
function [hdof ,sdof, vdof]= dof(Image,Lo_D,Hi_D)
  %Low Depth of Field
  %Get the dimensions of the image.  numberOfColorBands should be = 3.
  Image = imresize(Image, [64 64]);
  hsv_image = rgb2hsv(Image);
  hue_image = hsv_image(:,:,1);
  Saturation_image = hsv_image(:,:,2);
  Value_Image = hsv_image(:,:,3);
    
  [cA,~,~,cD] = dwt2(hue_image ,Lo_D,Hi_D);
  [cA,~,~,cD] = dwt2(cA,Lo_D,Hi_D);
  [~,cH,cV,cD] = dwt2(cA,Lo_D,Hi_D);
  w_hue = {cH,cV,cD};
  hdof = sumCo(w_hue);
    
  [cA,~,~,cD] = dwt2(Saturation_image ,Lo_D,Hi_D);
  [cA,~,~,cD] = dwt2(cA,Lo_D,Hi_D);
  [~,cH,cV,cD] = dwt2(cA,Lo_D,Hi_D);
  w_Saturation = {cH,cV,cD};
  sdof = sumCo( w_Saturation);
    
  [cA,~,~,cD] = dwt2(Value_Image ,Lo_D,Hi_D);
  [cA,~,~,cD] = dwt2(cA,Lo_D,Hi_D);
  [~,cH,cV,cD] = dwt2(cA,Lo_D,Hi_D);
  w_Value=  {cH,cV,cD};
  vdof = sumCo( w_Value);
end