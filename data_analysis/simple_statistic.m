function simple_statistic(name, value1, value2, value3)

x=value1;
SEM = std(x)/sqrt(length(x));               % Standard Error
ts = tinv([0.025  0.975],length(x)-1);      % T-Score
CI = mean(x) + ts*SEM;                      % Confidence Intervals
fprintf("%s CT_2CBCT21\t mean is %.3f, median is %.3f, std is %.3f, CI is [%.3f, %.3f]\n", name, mean(x), median(x), std(x), CI(1), CI(2));

x=value2;
SEM = std(x)/sqrt(length(x));               % Standard Error
ts = tinv([0.025  0.975],length(x)-1);      % T-Score
CI = mean(x) + ts*SEM;                      % Confidence Intervals
fprintf("%s CBCT01_2CBCT21\t mean is %.3f, median is %.3f, std is %.3f, CI is [%.3f, %.3f]\n", name, mean(x), median(x), std(x), CI(1), CI(2));


x=value3;
SEM = std(x)/sqrt(length(x));               % Standard Error
ts = tinv([0.025  0.975],length(x)-1);      % T-Score
CI = mean(x) + ts*SEM;                      % Confidence Intervals
fprintf("%s Pred_2CBCT21\t mean is %.3f, median is %.3f, std is %.3f, CI is [%.3f, %.3f]\n", name, mean(x), median(x), std(x), CI(1), CI(2));
fprintf("\n")