% Function to compute Color Harmony given an image.
% implmented from:
% Nishiyama, M.; Okabe, T.; Sato, I.; Sato, Y., "Aesthetic quality classification
% of photographs based on color harmony," Computer Vision and Pattern Recognition 
% (CVPR), 2011 IEEE Conference on , vol., no., pp.33,40, 20-25 June 2011

function HS = color_harmony(Image)
  HSVmap = rgb2hsv(Image);
  Count = imhist(HSVmap(:,1),100);
    
  %i type
  template1 = zeros(100,1);
  template1(1:5,:)=1;
  harmony_1= histintersection(Count,template1);
  conv1 = max(conv(template1,Count));
    
  %V type
  template2 = zeros(100,1);
  template2(1:26,:)=1;
  harmony_2= histintersection(Count,template2);
  conv2 =max( conv(template2,Count));
    
  %L type
  template3=zeros(100,1);
  template3(1:5,:)=1;
  template3(23:29,:)=1;
  harmony_3= histintersection(Count,template3);
  conv3 = max(conv(template3,Count));
  
  %I type
  template4=zeros(100,1);
  template4(1:5,:)=1;
  template4(49:54,:)=1;
  harmony_4= histintersection(Count,template4);
  conv4 = max(conv(template4,Count));
  
  % T type
  template5= zeros(100,1);
  template5(1:50,:)=1;
  harmony_5= histintersection(Count,template5);
  conv5 = max(conv(template5,Count));
    
  % Y type
  template6=zeros(100,1);
  template6(1:26,:)=1;
  template6(51:56,:)=1;
  harmony_6= histintersection(Count,template6);
  conv6 = max(conv(template6,Count));
    
  % X type
  template7=zeros(100,1);
  template7(1:25,:)=1;
  template7(50:75,:)=1;
  harmony_7= histintersection(Count,template7);
  conv7 = max(conv(template7,Count));

  % Pick Maximum Harmony Value as Harmony Score
  H = [harmony_1,harmony_2,harmony_3,harmony_4,harmony_5,harmony_6,harmony_7];
  CH = [conv1,conv2,conv3,conv4,conv5,conv6,conv7];
    
  HS = [max(H), max(CH)];
end