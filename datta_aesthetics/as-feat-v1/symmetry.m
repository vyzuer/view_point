%This function computes symmetry score for images
function [cost_lr cost_tb] = symmetry(I)
% choose the number of columns and rows you want to divide the image into
cols = 10;
rows = 10;

Ileft = I(:, 1:floor(size(I,2)/2), :); % Left half of the image
Iright = I(:, end:-1:ceil(size(I,2)/2), :); % Right half of the image
Itop = I(1:floor(size(I,1)/2),:,:); % Top Half
Ibottom = I(end:-1:ceil(size(I,1)/2),:,:); % Bottom Half

leftSq = {}; % Structure for storing patches and descriptors
rightSq = {}; % Structure for storing patches and descriptors
topSq = {}; % Structure for storing patches and descriptors
BottomSq = {}; % Structure for storing patches and descriptors
%%

colSize = floor(size(Ileft,2)/cols);
rowSize = floor(size(Ileft,1)/rows);

leftDescMat = []; % matrix for storing descriptors only

% loop over all patches in left image and get HOG and color histogram
% feature vector

for i = 1:rows
    
    for j = 1:cols
        
        % get patch at row 'i' and column 'j'
        leftSq{i}{j}.im = Ileft((i-1)*rowSize+1:i*rowSize,(j-1)*colSize+1:j*colSize,:);
        % get centroid of patch
        leftSq{i}{j}.cent = [i*rowSize - round(rowSize/2) j*colSize - round(colSize/2)];
        % get HOG feature vector from this image patch
        leftSq{i}{j}.hog = myHOG(leftSq{i}{j}.im);
        
        % get red channel of image patch
        rPatch = (leftSq{i}{j}.im(:,:,1));
        % convert the red channel of image patch into a vector
        rPatch = rPatch(:);
        % get histogram of the red image patch (ten bins)
        rHist = hist(round((rPatch(:)./256).*10),1:10);
        % normalize histogram 
        rHist = rHist./(norm(rHist) + 0.01);
        
        % get green channel of image patch
        gPatch = (leftSq{i}{j}.im(:,:,2));
        % vectorize green channel patch 
        gPatch = gPatch(:);
        % get histogram of green image patch (ten bins)
        gHist = hist(round((gPatch(:)./256).*10),1:10);
        % normalize histogram
        gHist = gHist./(norm(gHist) + 0.01);
        
        % get blue channel of image patch
        bPatch = (leftSq{i}{j}.im(:,:,1));
        % vectorize blue channel patch
        bPatch = bPatch(:);
        % get histogram of blue image patch (ten bins)
        bHist = hist(round((bPatch(:)./256).*10),1:10);
        % normalize histogram
        bHist = bHist./(norm(bHist) + 0.01);
        
        % concatenate histogram from each color channel into one vector
        leftSq{i}{j}.colorhist = [rHist gHist bHist];
        
        % store feature vector for image patch in new row of matrix
        % leftDescMat, feature vector consists of hog , color hist and
        % centroid and row, column information
        leftDescMat = [leftDescMat; [leftSq{i}{j}.hog' leftSq{i}{j}.colorhist leftSq{i}{j}.cent i j]];
%         leftDescMat = [leftDescMat; [leftSq{i}{j}.hog' leftSq{i}{j}.cent i j]];
        
    end
end
     
%%
rightDescMat = [];
% loop over all patches in right image and get HOG
for i = 1:rows
    
    for j = 1:cols
        
        % get patch at row 'i' and column 'j'
        rightSq{i}{j}.im = Iright((i-1)*rowSize+1:i*rowSize,(j-1)*colSize+1:j*colSize,:);
        % get centroid of patch
        rightSq{i}{j}.cent = [i*rowSize - round(rowSize/2) j*colSize - round(colSize/2)];
        % get HOG feature vector from this image patch
        rightSq{i}{j}.hog = myHOG(rightSq{i}{j}.im);
        
        % get red channel of image patch
        rPatch = (rightSq{i}{j}.im(:,:,1));
        % convert the red channel of image patch into a vector
        rPatch = rPatch(:);
        % get histogram of the red image patch (ten bins)
        rHist = hist(round((rPatch(:)./256).*10),1:10);
        % normalize histogram 
        rHist = rHist./(norm(rHist) + 0.01);
        
        % get green channel of image patch
        gPatch = (rightSq{i}{j}.im(:,:,2));
        % vectorize green channel patch 
        gPatch = gPatch(:);
        % get histogram of green image patch (ten bins)
        gHist = hist(round((gPatch(:)./256).*10),1:10);
        % normalize histogram
        gHist = gHist./(norm(gHist) + 0.01);
        
        % get blue channel of image patch
        bPatch = (rightSq{i}{j}.im(:,:,1));
        % vectorize blue channel patch
        bPatch = bPatch(:);
        % get histogram of blue image patch (ten bins)
        bHist = hist(round((bPatch(:)./256).*10),1:10);
        % normalize histogram
        bHist = bHist./(norm(bHist) + 0.01);
        
        % concatenate histogram from each color channel into one vector
        rightSq{i}{j}.colorhist = [rHist gHist bHist];
        
        % store feature vector for image patch in new row of matrix
        % rightDescMat, feature vector consists of hog , color hist and
        % centroid and row, column information
        rightDescMat = [rightDescMat; [rightSq{i}{j}.hog' rightSq{i}{j}.colorhist rightSq{i}{j}.cent i j]];
%         rightDescMat = [rightDescMat; [rightSq{i}{j}.hog' rightSq{i}{j}.cent i j]];
        
    end
end
%%
colSize2 = floor(size(Itop,2)/cols);
rowSize2 = floor(size(Itop,1)/rows);

topDescMat = []; % matrix for storing descriptors only

% loop over all patches in top image and get HOG and color histogram
% feature vector

for i = 1:rows
    
    for j = 1:cols
        
        % get patch at row 'i' and column 'j'
        topSq{i}{j}.im = Itop((i-1)*rowSize2+1:i*rowSize2,(j-1)*colSize2+1:j*colSize2,:);
        % get centroid of patch
        topSq{i}{j}.cent = [i*rowSize2 - round(rowSize2/2) j*colSize2 - round(colSize2/2)];
        % get HOG feature vector from this image patch
        topSq{i}{j}.hog = myHOG(topSq{i}{j}.im);
        
        % get red channel of image patch
        rPatch = (topSq{i}{j}.im(:,:,1));
        % convert the red channel of image patch into a vector
        rPatch = rPatch(:);
        % get histogram of the red image patch (ten bins)
        rHist = hist(round((rPatch(:)./256).*10),1:10);
        % normalize histogram 
        rHist = rHist./(norm(rHist) + 0.01);
        
        % get green channel of image patch
        gPatch = (topSq{i}{j}.im(:,:,2));
        % vectorize green channel patch 
        gPatch = gPatch(:);
        % get histogram of green image patch (ten bins)
        gHist = hist(round((gPatch(:)./256).*10),1:10);
        % normalize histogram
        gHist = gHist./(norm(gHist) + 0.01);
        
        % get blue channel of image patch
        bPatch = (topSq{i}{j}.im(:,:,1));
        % vectorize blue channel patch
        bPatch = bPatch(:);
        % get histogram of blue image patch (ten bins)
        bHist = hist(round((bPatch(:)./256).*10),1:10);
        % normalize histogram
        bHist = bHist./(norm(bHist) + 0.01);
        
        % concatenate histogram from each color channel into one vector
        topSq{i}{j}.colorhist = [rHist gHist bHist];
        
        % store feature vector for image patch in new row of matrix
        % leftDescMat, feature vector consists of hog , color hist and
        % centroid and row, column information
      topDescMat = [topDescMat; [topSq{i}{j}.hog' topSq{i}{j}.colorhist topSq{i}{j}.cent i j]];
%         leftDescMat = [leftDescMat; [leftSq{i}{j}.hog' leftSq{i}{j}.cent i j]];
        
    end
end
%%

bottomDescMat = [];
% loop over all patches in right image and get HOG
for i = 1:rows
    
    for j = 1:cols
        
        % get patch at row 'i' and column 'j'
        bottomSq{i}{j}.im = Ibottom((i-1)*rowSize2+1:i*rowSize2,(j-1)*colSize2+1:j*colSize2,:);
        % get centroid of patch
        bottomSq{i}{j}.cent = [i*rowSize2 - round(rowSize2/2) j*colSize2 - round(colSize2/2)];
        % get HOG feature vector from this image patch
        bottomSq{i}{j}.hog = myHOG(bottomSq{i}{j}.im);
        
        % get red channel of image patch
        rPatch = (bottomSq{i}{j}.im(:,:,1));
        % convert the red channel of image patch into a vector
        rPatch = rPatch(:);
        % get histogram of the red image patch (ten bins)
        rHist = hist(round((rPatch(:)./256).*10),1:10);
        % normalize histogram 
        rHist = rHist./(norm(rHist) + 0.01);
        
        % get green channel of image patch
        gPatch = (bottomSq{i}{j}.im(:,:,2));
        % vectorize green channel patch 
        gPatch = gPatch(:);
        % get histogram of green image patch (ten bins)
        gHist = hist(round((gPatch(:)./256).*10),1:10);
        % normalize histogram
        gHist = gHist./(norm(gHist) + 0.01);
        
        % get blue channel of image patch
        bPatch = (bottomSq{i}{j}.im(:,:,1));
        % vectorize blue channel patch
        bPatch = bPatch(:);
        % get histogram of blue image patch (ten bins)
        bHist = hist(round((bPatch(:)./256).*10),1:10);
        % normalize histogram
        bHist = bHist./(norm(bHist) + 0.01);
        
        % concatenate histogram from each color channel into one vector
        bottomSq{i}{j}.colorhist = [rHist gHist bHist];
        
        % store feature vector for image patch in new row of matrix
        % rightDescMat, feature vector consists of hog , color hist and
        % centroid and row, column information
        bottomDescMat = [bottomDescMat; [bottomSq{i}{j}.hog' bottomSq{i}{j}.colorhist bottomSq{i}{j}.cent i j]];
%         rightDescMat = [rightDescMat; [rightSq{i}{j}.hog' rightSq{i}{j}.cent i j]];
        
    end
end
%%

%Get distances between all descriptors of individual image patches in both image halves
distMat_lr =distChiSq (leftDescMat(:,1:end-4),rightDescMat(:,1:end-4)); % rows of distMat will correspond to first argument of pdist2

%Get corresponding patches
matches_lr = munkres(distMat_lr); % each index is the column of distMat matched to that row index

% Get column and row numbers of each patch pair in both halves
idsLeft = leftDescMat(:,end-1:end);
idsRight = rightDescMat(matches_lr,end-1:end); % rows of rightDescMat are indexed by "matches", 
% so each row will hold the location of the right image patch that was matched with the image patch 
% from Ileft at this row (Ileft rows have original order) 

% Get L1 distance of correspondigng patches
L1dists = sum(abs(idsLeft - idsRight),2);

% Final cost
cost_lr = sum(L1dists)/(rows*cols);
%%
% Get distances between all descriptors of individual image patches in both image halves
distMat_tb = distChiSq(topDescMat(:,1:end-4),bottomDescMat(:,1:end-4)); % rows of distMat will correspond to first argument of pdist2

% Get corresponding patches
matches_tb = munkres(distMat_tb); % each index is the column of distMat matched to that row index

% Get column and row numbers of each patch pair in both halves
idstop = topDescMat(:,end-1:end);
idsbottom = bottomDescMat(matches_tb,end-1:end); % rows of rightDescMat are indexed by "matches", 
% so each row will hold the location of the right image patch that was matched with the image patch 
% from Ileft at this row (Ileft rows have original order) 

% Get L1 distance of correspondigng patches
L1dists = sum(abs(idstop - idsbottom),2);
% Final cost
cost_tb = sum(L1dists)/(rows*cols);

  end