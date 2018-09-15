Total= 47.380508849;
Alpha_opt= 0.75;
ratio = 0.3;
d5= 13.815158896;
d0 = 11.84512721225;
c1 = 1.9849915274751876e-07^0.5;
c2 = 2.043973041652856e-07 ^0.5;
c3 = 3.513725392209836e-07^0.5;
c4 = 2.361075976051944e-05^0.5;
pkg load io

function Problem1(Alpha,ratio)
    if nargin <2
        ratio = 0.3
    end
    if nargin <1
        Alpha = 0.75
    end
    data = xlsread('CUMCM-2018-Problem-A-Chinese-Appendix.xlsx',2);
    X = data(:,1);
    Y = data(:,2);
    a = 21.720222740795744/(Alpha-(48.08-37)/(75-37));
    x = (48.08-37)/(75-37) * a;
    fprintf('a = %.5d\t',a)
    fprintf('x = %.5d\t',x)
    fprintf('alpha = %.5d\t',Alpha)
    Predict = Solution(x,a,37,75,Alpha,X);
    fprintf('误差的绝对值的平均值为 %8.5d\n',Mean_L1_error(Predict,Y));
    plot(X,Predict,X,Y)
    legend('Predicted','Ground Truth')
end