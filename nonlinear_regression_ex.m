%% Estimate theta
clc
clear all
close all

dP=[245182
    877284
    394215
    611533
    248648];

m=[2204.5
    901.1
    1675.9
    1218.0
    2202.7];

%% initial guess through transformation and linear regression
n=length(dP);
Y = 1./m;
X = [ones(n,1) dP];
thetahat = inv(X'*X)*(X'*Y);

rho = 1000; g = 9.81;
W0 = 1/(thetahat(2)*rho);
h0 = thetahat(1)/(thetahat(2)*rho*g);
fprintf('initial guess using transformation and linear regression:\n w0= %f,  h0= %f \n \n', W0, h0);


%% nonlinear least square
theta(:,1) = [W0;h0];

Y = m;

flag = true;
i = 1; % iteration number
while(flag) % iteration stop condition
   F(:,i) = theta(1,i)./(dP./rho + g*theta(2,i));
   Z(:,i) = Y-F(:,i); % Evaluate Z
   D(:,:,i) = [1./(dP./rho + g*theta(2,i))    -theta(1,i).*g./(dP./rho + g.*theta(2,i)).^2];
   dtheta(:,i) = inv(D(:,:,i)'*D(:,:,i))*D(:,:,i)'*Z(:,i); % Estimate delta theta
   theta(:,i+1) = theta(:,i) + dtheta(:,i); % Calculate estimate of theta
   if(norm(dtheta(:,i))<0.001)
       flag = false;
   end
   i = i+1;
end

theta

%% Confidence intervals
n = size(Y,1);
p = 2;
k = 2;
F_final = theta(1,end)./(dP./rho + g*theta(2,end));
Z_final = Y - F_final;
D_final = [1./(dP./rho + g*theta(2,end))    -theta(1,end).*g./(dP./rho + g.*theta(2,end)).^2];
Z_pred = D_final*dtheta(:,end);
epsilon = Z_final - Z_pred;
sigma2 = epsilon'*epsilon/(n-p);
C = inv(D_final'*D_final);
t_alpha = icdf('t',1-0.05/2,n-p);
% CI on dtheta
CI_l1 = theta(1,end) - t_alpha*sqrt(C(1,1)*sigma2);
CI_u1 = theta(1,end) + t_alpha*sqrt(C(1,1)*sigma2);
CI_l2 = theta(2,end) - t_alpha*sqrt(C(2,2)*sigma2);
CI_u2 = theta(2,end) + t_alpha*sqrt(C(2,2)*sigma2);

% CI on W and h
fprintf('confidence interval for parameters:\n %f <= w <= %f \n  %f <= h <= %f\n ',  CI_l1,  CI_u1,  CI_l2,   CI_u2);

%% CI for mean response at a single point
x0 = 0.5e6;  %x0
mu_m = theta(1,end)./(x0./rho + g*theta(2,end))
D_x0 = [1./(x0./rho + g*theta(2,end))    -theta(1,end).*g./(x0./rho + g.*theta(2,end)).^2];
CI_mu = [mu_m - t_alpha*sqrt(D_x0*C*D_x0'*sigma2)      mu_m + t_alpha*sqrt(D_x0*C*D_x0'*sigma2)]

%% CI for prediction for all points
CI_y0 = zeros(n,2);
for i=1:n
    x0 = dP(i);
    y0 = theta(1,end)./(x0./rho + g*theta(2,end));
    D_x0 = [1./(x0./rho + g*theta(2,end))    -theta(1,end).*g./(x0./rho + g.*theta(2,end)).^2];
    CI_y0(i,:) = [y0 - t_alpha*sqrt((1+D_x0*C*D_x0')*sigma2)    y0 + t_alpha*sqrt((1+D_x0*C*D_x0')*sigma2)];
end


%% plot
ypred = F_final;
Res = Y - F_final;
figure
plot(dP,m,'k*'); hold on
xlabel('\Delta P'); ylabel('m');
[sorted_dP, sortID] = sort(dP);  % to connnect the points, need data sorting
plot(dP(sortID), ypred(sortID),'-go'); hold on
plot(dP(sortID), CI_y0(sortID,1), 'r-'); hold on
plot(dP(sortID), CI_y0(sortID,2), 'r-');
legend('data','fitted model','CI for prediction');


%% residual analysis
% standardize residual
d = Res/sqrt(sigma2);
figure
plot(dP, d, 'ko'); hold on
plot([min(dP) max(dP)], [-2 -2], 'r--'); plot([min(dP) max(dP)], [2 2], 'r--'); ylim([-3.5, 3.5])
xlabel('\DeltaP (regressor)'); ylabel('Standardized residual'); title('Residual plot')

figure
plot(ypred, d, 'ko'); hold on
plot([min(ypred) max(ypred)], [-2 -2], 'r--'); plot([min(ypred) max(ypred)], [2 2], 'r--'); ylim([-3.5, 3.5])
xlabel('$$\hat{m}$$ (predicted response)','Interpreter','Latex'); ylabel('Standardized residual'); title('Residual plot')

figure
plot(1:n, d, 'ko'); hold on
plot(1:n, -2*ones(1,n), 'r--'); plot(1:n, 2*ones(1,n), 'r--'); ylim([-3.5, 3.5])
xlabel('Sample ID'); ylabel('Standardized residual'); xlim([0.5 n+0.5]); title('Residual plot')


