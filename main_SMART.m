%--------------------------------------------------------------------------
% Corresponding author: Qi Zhang
% Department of Applied Mathematics and Statistics,
% Stony Brook University, Stony Brook, NY 11794-3600
% Email: zhangqi{dot}math@gmail{dot}com
%--------------------------------------------------------------------------
% 1. The SMART algorithm by Qi Zhang and Jiaqiao Hu [1] is implemented for
% solving single-objective box-constrained expensive stochastic
% optimization problems.
% 2. In this implementation, the algorithm samples candidate solutions from 
% a sequence of independent multivariate normal distributions that recursively 
% approximiates the corresponding Boltzmann distributions [2].
% 3. In this implementation, the surrogate model is constructed by the 
% radial basis function (RBF) method [3].
%--------------------------------------------------------------------------
% REFERENCES
% [1] Qi Zhang and Jiaqiao Hu (2019): Actor-Critic Like Stochastic Adaptive Search
% for Continuous Simulation Optimization. Submitted to Operations Research,
% under review.
% [2] Jiaqiao Hu and Ping Hu (2011): Annealing adaptive search,
% cross-entropy, and stochastic approximation in global optimization.
% Naval Research Logistics 58(5):457-477.
% [3] Gutmann HM (2001): A radial basis function method for 
% global optimization. Journal of Global Optimization 19:201-227.
%--------------------------------------------------------------------------
% This program is a free software.
% You can redistribute and/or modify it. 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY, without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
%--------------------------------------------------------------------------
clearvars; close all;
%% PROBLEM SETTING
% All test functions H(x) are in dimension 20 (d=20)
% with box-constrain [-10,10]^d
% and scaled to have an optimal(max) objective value -1.
% The observation noise z(x) is normally distributed with a zero mean
% and a standard deviation that is proportional to |H(x)|, i.e.,
% z(x)~N(0,(noise_std*|H(x)|)^2).
d=20; % dimension of the search region
left_bound=-10; % left bound of the search region
right_bound=10; % right bound of the search region
optimal_objective_value=-1; % optimal objective value
noise_std=0.1;
%--------------------------------------------------------------------------
% The test functions can be chosen from 
% 1. 'sumSquares'
% 2. 'bohachevsky'
% 3. 'cigar'
% 4. 'ackley'
% 5. 'qing'
% 6. 'styblinskiTang'
% 7. 'pinter'
fcn_name='sumSquares';
%--------------------------------------------------------------------------

%% HYPERPARAMETERS
budget=2000; % total number of function evaluations assigned
warm_up=5*d; % the number of function evaluations used for warm up 
% alpha: learning rate for updating the mean parameter function m(theta)
% alpha=1/(k-warm_up+ca)^gamma0
ca=100; gamma0=.51;
% beta: learning rate for updating the performance estimation Hk(x)
% beta=1./Nk.^gamma1 where Nk is the number of times that the shrink ball
% being hit
gamma1=0.99; 
% rk: the shrinking ball radius
% rk=cr/log(k-warm_up+1)^gamma2
% cr=(right_bound-left_bound)*0.001*sqrt(d)*log(budget)^1.01
gamma2=1.01;
cr=(right_bound-left_bound)*0.001*sqrt(d)*log(budget)^1.01;
% exploration: sampling exploration factor
exploration=0;

%% INITIALIZATION
% Initial mean of the sampling distribution
mu_old=left_bound+(right_bound-left_bound)*rand(d,1);
% Initial variance of the sampling distribution
var_old=((right_bound-left_bound)/2)^2*ones(d,1);
% Calculating the initial mean parameter function value
[eta_x_old,eta_x2_old]=...
    truncated_mean_para_fcn(left_bound,right_bound,mu_old,var_old);

%% RECORDS
c_best_H=[]; % current best true objective value found
H=[]; % true objective values
h=[]; % noisy observations
Hk=[]; % performance estimations
Nk=[]; % the number of times shrinking balls being hit

%% WARM UP PERIOD
% A warm up period is performed in order to get a robust algorithm
% performance. The idea is using the sobol set to construct a trustable
% surrogate model at the beginning.
fprintf('Warm up begins \n');
tic; % count warm up time

sobol_all=sobolset(d);
Lambda=net(sobol_all,warm_up); % Lambda: records all sampled solutions
Lambda=left_bound+(right_bound-left_bound)*Lambda; Lambda=Lambda';
Nk(1:warm_up)=ones(1,warm_up);
% D: initial distance matrix for calculating the surrogate model's weights
D=zeros(warm_up,warm_up);
for i=1:warm_up-1
    for j=i+1:warm_up
        D(i,j)=norm(Lambda(:,i)-Lambda(:,j))^3;
    end
end
D=D+D';
% We do NOT need to implement the shrinking ball strategy here since the
% observations in the warm up period are far enough.
H(1:warm_up)=feval(fcn_name,Lambda);
c_best_H(1:warm_up)=max(H(1:warm_up));
h(1:warm_up)=H+abs(H).*normrnd(0,noise_std,1,warm_up);
Hk(1:warm_up)=h(1:warm_up);
% weight: the coefficients of the surrogate model
weight=D\Hk';
fprintf('Warm up ends \n');
tWarmUp=toc; % count warm up time
fprintf('Warm up takes %8.4f seconds \n',tWarmUp);

%% MAIN LOOP
fprintf('Main loop begin.\n');
tic; % count main loop time
k=warm_up; % iteration counter
num_evaluation=warm_up; % budget consumption
while num_evaluation+1<=budget
    %% PROGRESS REPORT
    if mod(k,100)==0
        fprintf('iter: %5d, eval: %5d, cur best: %8.4f, true optimum: %8.4f \n',...
            k,num_evaluation,c_best_H(k),optimal_objective_value);
    end
    k=k+1;
    
    %% ADAPTIVE HYPERPARAMETERS
    % alpha: learning rate for updating the mean parameter function
    alpha=1/(k-warm_up+ca)^gamma0;
    % rk: shrinking ball radius
    rk=cr/log(k-warm_up+1)^gamma2;
    % t: annealing temperature
    t=.1*abs(c_best_H(end))/log(k-warm_up + 1);
    % numNumInt: number of points for numerical integration
    numNumInt=max(100,floor((k-warm_up)^.5));
    
    %% SAMPLING
    % Given the sampling parameter theta_old=(mu_old,var_old),
    % generate a sample x from the independent multivariate normal density.
    distribution_choice=rand();
    if distribution_choice<exploration
        % the exploration part from the uniform distribution
        x_sample=left_bound+(right_bound-left_bound)*rand(d,1);
    else
        x_sample=normt_rnd(mu_old,var_old,left_bound,right_bound);
    end
    Lambda(:,k)=x_sample;
    Nk(k)=1;
    
    % Record the current best true objective value.
    H(k)=feval(fcn_name,x_sample);
    c_best_H(k)=max(c_best_H(k-1),H(k));
    
    %% PERFORMANCE ESTIMATIONS
    % h: noisy observations
    h(k)=H(k)+abs(H(k))*normrnd(0,noise_std);
    num_evaluation=num_evaluation+1;
    % I_xk: indices that Lambda(:,I_xk) hits the ball B(xk,rk)
    I_xk=find(vecnorm( Lambda(:,1:k-1)-Lambda(:,k))<rk);
    % Hk(xk): initial estimation
    Hk(k)=mean(cat(2,Hk(I_xk),h(k)));
    % Hk(1:k-1): update performance estimations if xi hits the current ball
    Nk(I_xk)=Nk(I_xk)+1;
    beta=1./Nk.^gamma1;
    Hk(I_xk)=(1-beta(I_xk)).*Hk(I_xk)+beta(I_xk)*h(k);
    
    %% SURROGATE MODELING
    % Given Hk, find the new surrogate model based on sampled solutions
    % Cubic model: S_k(x)=\sum_{i=1}^{k} weight(i)*||x-xi||^3
    % Surrogate model is a coefficient vector named weight
    temp=zeros(k,k);
    temp(1:k-1,1:k-1)=D;
    temp(end,1:end-1) = vecnorm(Lambda(:,1:k-1)-Lambda(:,k)).^3;
    temp(1:end-1,end) = temp(end,1:end-1)';
    D=temp;
    weight=D\Hk(1:k)';
    
    %% SAMPLING PARAMETER UPDATING
    % Given the surrogate model (weight),
    % update the sampling parameter mu, var.
    
    % Using numerical integration to find Ex Ex2 based on the surrogate
    % model.
    % 1. Generate numNumInt points from the sampling distribution.
    % 2. Calculate exp_surrogate_num_int_pts based on these samples.
    x_num_int=...
        normt_rnd(mu_old*ones(1,numNumInt),var_old*ones(1,numNumInt),...
        left_bound,right_bound);
    % exp_surrogate_num_int_pts=[..,exp(\sum_j w || xi-xj ||^3),..]
    % exponential of surrogate model at numerical integration points
    exp_surrogate_num_int_pts=...
        esnip_fcn(x_num_int,numNumInt,weight,Lambda(:,1:k),...
        t,mu_old,var_old,left_bound,right_bound);
  
    % scale
    exp_surrogate_num_int_pts=...
        exp_surrogate_num_int_pts/max(exp_surrogate_num_int_pts);
    
    int_exp_surrogate=sum(exp_surrogate_num_int_pts);
    int_exp_surrogate_x=sum(x_num_int.*exp_surrogate_num_int_pts,2);
    int_exp_surrogate_x2=sum(x_num_int.^2.*exp_surrogate_num_int_pts,2);
    
    EX = int_exp_surrogate_x/int_exp_surrogate;
    EX2 = int_exp_surrogate_x2/int_exp_surrogate;
    % Update the mean parameter function.
    eta_x_new=eta_x_old+alpha*(EX-eta_x_old);
    eta_x2_new=eta_x2_old+alpha*(EX2-eta_x2_old);
    
    [mu_new,var_new]=...
        truncated_inv_mean_para_fcn(left_bound,right_bound,...
        mu_old,var_old,eta_x_new,eta_x2_new);

    %% UPDATING
    eta_x_old=eta_x_new;
    eta_x2_old=eta_x2_new;
    
    mu_old=mu_new; 
    var_old = var_new;
    
    %% VISUALIZATION
    if mod(k,10)==0
        plot(c_best_H,'r-o'); % current best
        hold on
        cur_size=size(c_best_H);
        optimal_line=optimal_objective_value*ones(cur_size(2),1);
        plot(optimal_line,'k:','LineWidth',5); % true optimal value
        xlabel('Number of function evaluations')
        ylabel('Objective function value')
        title(sprintf('<%s function>   Iteration: %5d  Evaluation: %5d',fcn_name,k,num_evaluation));
        legend('SMART','True optimal value','Location','southeast');
        ylim([c_best_H(1)*1.1 -0.8]);
        grid on
        drawnow;
    end
end

%% FINAL REPORT
fprintf('iter: %5d, eval: %5d, cur best: %8.4f, true optimum: %8.4f \n',...
    k,num_evaluation,c_best_H(k),optimal_objective_value);
fprintf('Main loop ends \n');
tMainLoop=toc; % count main loop time
fprintf('Main loop takes %8.4f seconds \n',tMainLoop);
