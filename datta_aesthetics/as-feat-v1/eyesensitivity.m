function [ Pleasing_Score ] = Eye_Sensitivity( Image )
% Eye Sensitivity
HSVmap = rgb2hsv(Image);
Hue_map =  HSVmap (:,1);
Counts = hist(Hue_map,360);
%Intializing each color bin
red_counter = 0;
Orange_counter = 0;
yellow_counter = 0;
Green_yellow_counter = 0;
Green_counter = 0;
Green_cyan_counter = 0;
cyan_counter = 0;
blue_cyan_counter = 0;
blue_counter = 0;
violet_blue_counter = 0;
violet_counter = 0;
%Counting the number of pixels related to each color bin
for a=1:30
    if Counts(a)>0
        red_counter = red_counter+1;
    end
end
for a=31:60
    if Counts(a)>0
        Orange_counter = Orange_counter+1;
    end
end
for a=61:90
    if Counts(a)>0
        yellow_counter = yellow_counter+1;
    end
end
for a=91:120
    if Counts(a)>0
        Green_yellow_counter = Green_yellow_counter +1;
    end
end

for a=121:150
    if Counts(a)>0
        Green_counter = Green_counter +1;
    end
end
for a=151:180
    if Counts(a)>0
        Green_cyan_counter = Green_cyan_counter +1;
    end
end

for a=181:210
    if Counts(a)>0
        cyan_counter = cyan_counter +1;
    end
end
for a=211:240
    if Counts(a)>0
        blue_cyan_counter  = blue_cyan_counter +1;
    end
end

for a=241:270
    if Counts(a)>0
        blue_counter=blue_counter+1;
    end
end
for a=271:300
    if Counts(a)>0
        violet_blue_counter = violet_blue_counter +1;
    end
end
for a=301:360
    if Counts(a)>0
        violet_counter =  violet_counter +1;
    end
end

Colormap = [red_counter, Orange_counter, yellow_counter,Green_yellow_counter, Green_counter, Green_cyan_counter, cyan_counter,blue_cyan_counter,blue_counter, violet_blue_counter, violet_counter];
Color_Scores = [0.008;0.6;0.8;0.9;1;0.7;0.5;0.4;0.08;0.06;0.01];
Pleasing_Score = Colormap *Color_Scores;
end

