% This codesx uses s3 map to compute statistical features from sharpness for
% an image
function [ s3_mean,s3_std, s3_min, s3_max] = sharpness(Image)
    [s3] = s3_map(double(rgb2gray(Image)));
    Vectorized_s3 =  reshape(s3,1,[]);
    sorted_s3 = sort(Vectorized_s3);
    num_top = floor(size(sorted_s3,2)*0.01);
    vector_top = sorted_s3((size(sorted_s3,2)-num_top):end);
    s3_mean= mean(vector_top );
    s3_std = std(vector_top );
    s3_min = min(vector_top );
    s3_max = max(vector_top );
    
 end

