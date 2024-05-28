clear all
    data = [240	25	24	91	100
			236	31	21	90	95
			270	45	24	88	110
			274	60	25	87	88
            301	65	25	91	94
            316	72	26	94	99
            300	80	25	87	97
            296	84	25	86	96
            267	75	24	88	110
            276	60	25	91	105
            288	50	25	90	100
            261	38	23	89	98];
Y = data(:,1); % formulate Y
X = [ones(size(Y,1),1) data(:,2:5)]; % formulate X

%% Estimate hat theta
thetahat = inv(X'*X)*X'*Y % estimate theta
Yhat = X*thetahat; % predict the responses
epsilon = Y - Yhat; % residuals

%% ANOVA
n = size(Y,1);
k = 4;
p = 5;
SSE = epsilon'*epsilon; % SSE
SSR = thetahat'*X'*Y - sum(Y)^2/n; % SSR
SST = (Y - mean(Y).*ones(n,1))'*(Y - mean(Y).*ones(n,1)); % SST
MSE = SSE/(n-p); % MSE
MSR = SSR/k; % MSR
F0 = MSR/MSE % Test statistic F0
f_alpha = icdf('f',1-0.05,k,n-p) % f value

%% CI calcuation
C = inv(X'*X); % C matrix
t_alpha = icdf('t',1-0.05/2,n-p); % t value when alpha = 0.05
sigma2 = epsilon'*epsilon/(n-p); % estimate of sigma square
CI_l = zeros(p,1);
CI_u = zeros(p,1);
for i=1:p
    CI_l(i) = thetahat(i) - t_alpha*sqrt(C(i,i)*sigma2); % lower bound on theta_i
    CI_u(i) = thetahat(i) + t_alpha*sqrt(C(i,i)*sigma2); % upper bound on theta_i
end
CI_l
CI_u

%% Prediction and its confidence interval
x0 = [1 75 24 90 98]'; % the x
ypred = x0'*thetahat % predicted power consumption

CI_ypred_l = ypred - t_alpha*sqrt(sigma2*(1+x0'*C*x0))
CI_ypred_u = ypred + t_alpha*sqrt(sigma2*(1+x0'*C*x0))

%% Calculate R2 and R2_adj
R2 = SSR/SST
R2_adj = 1 - (SSE/(n-p))/(SST/(n-1))

%% Residual tests
d = epsilon./sqrt(sigma2);
figure(1)
plot(d,'o')
xlabel('sample number')
ylabel('standardized residual')
figure(2)
plot(Yhat,d,'o')
xlabel('predicted response')
ylabel('standardized residual')
figure(3)
plot(X(:,2), d,'o')
xlabel('x_1')
ylabel('standardized residual')
figure(4)
normplot(d);
