function desc = myHOG(Im)

B = 9;
hx = [-1,0,1];
hy = -hx';
grad_xr = imfilter(double(Im),hx);
grad_yu = imfilter(double(Im),hy);
angles=atan2(grad_yu,grad_xr);
magnit=((grad_yu.^2)+(grad_xr.^2)).^.5;
v_angles=angles(:);    
v_magnit=magnit(:);
K=max(size(v_angles));
%assembling the histogram with 9 bins (range of 20 degrees per bin)
bin=0;
H2=zeros(B,1);
for ang_lim=-pi+2*pi/B:2*pi/B:pi
    bin=bin+1;
    for k=1:K
        if v_angles(k)<ang_lim
            v_angles(k)=100;
            H2(bin)=H2(bin)+v_magnit(k);
        end
    end
end

H2=H2/(norm(H2)+0.01);
desc = H2;