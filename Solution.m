function [ans] = Solution(x,a,T1,T2,Alpha,t)
    temp = 1:1000;
    mask1 = 2*(T2-T1)/pi./temp;
    mask2 = cos((temp*pi*Alpha));
    mask3 = sin(temp*pi*x/a);
    temp2 = -reshape(t,length(t),1) * (temp.^2) * pi^2/a^2;
    mask4 = exp(temp2);
    ans = repmat(mask1.*mask2.*mask3,length(t),1) .* mask4;
    ans = sum(ans,2);
    ans = ans + (T2-T1)*x/a+T1;
end

