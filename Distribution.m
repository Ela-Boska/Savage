
function Distribution()
    pkg load io
    alpha = 0.75;
    data = xlsread('CUMCM-2018-Problem-A-Chinese-Appendix.xlsx',2);
    thickness = 0.6+6+3.6+5;
    ts = data(:,1);
    clear data;
    xs = linspace(0,thickness,500);
    ans = zeros(length(ts)+1,500);
    ans(1,:) = xs;
    Xs = Transform(xs);
    for i = 1:500
        x = Xs(i);
        ans(2:length(ts)+1,i) = Solution(x,47.380508849,37,75,0.75,ts);
        disp(i)
    end
    xlswrite('Problem1.xlsx',ans);

end

