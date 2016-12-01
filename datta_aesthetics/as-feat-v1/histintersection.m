%% Histogram Intersection
function [ intersection ] = histintersection( hist1, hist2)
minsum = 0;
m = size(hist1);
for i=1 : m 
        minsum =minsum+ min(hist1(i), hist2(i));
    
end
intersection = minsum / min(sum(sum(hist1)), sum(sum(hist2)));
end