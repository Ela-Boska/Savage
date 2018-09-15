function [ans] = Transform(s)
    k1 = 1.9849915274751876e-07;
    k2 = 2.043973041652856e-07;
    k3 = 3.513725392209836e-07;
    k4 = 2.361075976051944e-05;
    c1 = 1000*k1^0.5;
    c2 = 1000*k2^0.5;
    c3 = 1000*k3^0.5;
    c4 = 1000*k4^0.5;
    thick0 = 47.380508849*0.75-21.720222740795744;
    thick1 = 5/c4 +thick0;
    thick2 = thick1+3.6/c3;
    thick3 = thick2+6/c2;
    mask1 = (s>=0) .* (s<=5);
    mask2 = (s>5) .* (s<=8.6);
    mask3 = (s>8.6) .* (s<=14.6);
    mask4 = (s>14.6) .* (s<=15.2);
    ans1 = mask1.*(s/c1+thick0);
    ans2 = mask2.*((s-5)/c2+thick1);
    ans3 = mask3.*((s-8.6)/c3+thick2);
    ans4 = mask4.*((s-14.6)/c4+thick3);
    ans = ans1+ans2+ans3+ans4;
end