%% Summation of Coeeficients( used in depth of field function)
 function [ depth_of_field  ] = SumCo(w)
 Sum_w(1:3)=0;
    for indx=1:3
        for i=3:6
            for j=3:6
                s = w{indx}(i,j);
                Sum_w(indx) = Sum_w(indx )+s;
            end
        end
    end
    for index=1:3
        Overal_Sum(index) = sum(sum(w{index}));
    end
   depth_of_field =  sum(Sum_w)/sum(Overal_Sum)+eps;
 end