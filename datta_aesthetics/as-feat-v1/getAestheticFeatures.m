function getAestheticFeatures (imfile, outfile)
  asf = struct();
  % Required for Depth-of-field based feature computation
  [Lo_D, Hi_D, ~, ~] = wfilters('haar');
  
  % Basic pre-processing
  try
    img = imread(imfile);
  catch
    fprintf(1, 'ERR: Failed to read %s', char(imfile));
    fa =[];
    fp = [];
    return;
  end
  
  [h, w, c] = size(img);
  a = min(w,h);
  % If image is too small, perform interpolation
  if (a<60) 
    img = imresize(img,60/a,'bicubic'); 
  end; 
  % If image is grayscale
  if (c==1)
    img=gray2rgb(img);
  end;
  % update width or height
  [h, w,  ~] = size(img);
  wd = w-mod(w,3);
  ht = h-mod(h,3);
  img = img(1:ht,1:wd,:);
  
  % Creating Blocks for cell level features
  ht3 = floor(ht/3);
  wd3 = floor(wd/3);
  % Grid line vectors
  hv = [0 floor(ht/3) 2*floor(ht/3) ht];
  wv = [0 floor(wd/3) 2*floor(wd/3) wd];
  S = floor(ht/3)*floor(wd/3);
 
  % Required for positional features - distance from 4 stress points
  spts = [wd3, ht3; 2*wd3, ht3; wd3, 2*ht3; 2*wd3, 2*ht3];
  
  % Find saliency map : for dominant object based features
  smap = saliency(img,0) ;
  level_smap= graythresh(smap);
  BW_smap = im2bw(smap,level_smap);
  area = regionprops(BW_smap,'Area');
  
  % normalizing area with respect to block size
  fa  = max([area(1:end).Area])/S;
  
  % Get centroid of convex hull
  c_hull = bwconvhull(BW_smap);
  cent = regionprops(c_hull ,'Centroid');
  % find distances of centroid wrt 4 stress points
  fp = sqrt(sum((repmat(cent.Centroid,[4 1])-spts).^2,2))/(ht*wd);
  
  % Dark Channel specfic requirements
  srgb2lab = makecform('srgb2lab');
  delta = 0.0001;
  patch_size = 10;
  I = double(img) ./ 255;
  % Make grayscales to color
  if numel(size(I)) == 2
    [x y] = size(I);
    tmpI = zeros(x,y,3);
    for c = 1:3
      tmpI(:,:,c) = I;
    end
    I = tmpI;
  end
  J = makeDarkChannel(I,patch_size);
  L = applycform(I, srgb2lab);
  
  
  % compute stats on dark-channel, Luminosity
  cellcount = 1;
  for i=1:size(hv,2)-1
    for j=1:size(wv,2)-1
      % Regular Image region in the grid
      den=I(hv(i)+1:hv(i+1), wv(j)+1:wv(j+1),:);
      
      % Compute dark channel feature speacific to grid element
      num=J(hv(i)+1:hv(i+1), wv(j)+1:wv(j+1));
      asf(cellcount).dc = sum(sum(num./(sum(den,3)+0.0000001)))/S;
      
      % Compute Luminosity feature speacific to grid element
      lum =L(hv(i)+1:hv(i+1), wv(j)+1:wv(j+1), 1);
      asf(cellcount).lc = exp(sum(sum(log(delta*ones(size(lum)) + lum)))/S);
            
      %S3 Sharpness
      [shp_avg, shp_var, shp_min, shp_max] = sharpness(den);
      asf(cellcount).shp = [shp_avg, shp_var, shp_min, shp_max];
      
      % Symmetry
      [lrsym, tdsym] = symmetry(den);
      asf(cellcount).sym = [lrsym , tdsym];
      
      % Low Depth of Field
      asf(cellcount).dof = dof(den,Lo_D,Hi_D);
            
      % White Balance
      asf(cellcount).wb = whitebalance(den,S);
             
      % Colorfulness
      asf(cellcount).cf = colorfulness(den);
    
      % Color Harmony
      asf(cellcount).ch = charmony(den);
      
      % Eye Sensitivity
      asf(cellcount).es = eyesensitivity(den);
      cellcount = cellcount+1;
    end
  end
  % Finally concatenate all descriptors
  desc = [fa, fp', [asf.dc], [asf.lc], [asf.shp], [asf.sym], [asf.dof], ...
     [asf.wb], [asf.cf], [asf.ch], [asf.es]];
 
  dlmwrite(outfile, desc, '-append', 'delimiter', ' ');
return
