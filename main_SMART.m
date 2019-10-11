%--------------------------------------------------------------------------
% Corresponding author: Qi Zhang
% Department of Applied Mathematics and Statistics,
% Stony Brook University, Stony Brook, NY 11794-3600
% Email: zhangqi{dot}math@gmail{dot}com
%--------------------------------------------------------------------------
% 1. The SMART algorithm by Qi Zhang and Jiaqiao Hu [1] is implemented for
% solving single-objective box-constrained expensive stochastic
% optimization problems
% 2. In this implementation, the algorithm samples candidate solutions from 
% a sequence of independent multivariate normal distributions that recursively 
% approximiates the corresponding Boltzmann distributions [2]
% 3. In this implementation, the surrogate model is constructed by the 
% radial basis function (RBF) method [3]
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
% and scaled to have an optimal(max) objective value -1
% The observation noise z(x) is normally distributed with a zero mean
% and a standard deviation that is proportional to |H(x)|, i.e.,
% z(x)~N(0,(noise_std*|H(x)|)^2)
d=20; % dimension of the search region
left_bound=-10; % left bound of the search region
right_bound=10; % right bound of the search region
optimal_objective_value=-1; % optimal objective value
noise_std=0.1; % standard deviation of the observation noise
%--------------------------------------------------------------------------
% The test functions can be chosen from 
% 1. 'sumSquares_test_fcn'
% 2. 'bohachevsky_test_fcn'
% 3. 'cigar_test_fcn'
% 4. 'ackley_test_fcn'
% 5. 'qing_test_fcn'
% 6. 'styblinskiTang_test_fcn'
% 7. 'pinter_test_fcn'
fcn_name='sumSquares_test_fcn';
%--------------------------------------------------------------------------

%% HYPERPARAMETERS (problem-dependent)
budget=2000; % total number of function evaluations assigned
warm_up=5*d; % number of function evaluations used for warm up 
% alpha: learning rate for updating the mean parameter function m(theta)
% alpha=1/(k-warm_up+ca)^gamma0
gamma0=.51; ca=100;
% beta: learning rate for updating the performance estimation Hk(x)
% beta=1./Nk.^gamma1 where Nk is the number of times that the shrink ball
% being hit
gamma1=0.99; 
% rk: shrinking ball radius
% rk=cr/log(k-warm_up+1)^gamma2
% cr=(right_bound-left_bound)*0.001*sqrt(d)*log(budget)^1.01
gamma2=1.01; cr=(right_bound-left_bound)*0.001*sqrt(d)*log(budget)^1.01;

%% INITIALIZATION
% Initial mean of the sampling distribution
mu_old=left_bound+(right_bound-left_bound)*rand(d,1);
% Initial variance of the sampling distribution
var_old=((right_bound-left_bound)/2)^2*ones(d,1);
% Calculating the initial mean parameter function value
[eta_x_old,eta_x2_old]=...
    truncated_mean_para_fcn(left_bound,right_bound,mu_old,var_old);

%% WARM UP PERIOD
% A warm up period is performed in order to get a robust algorithm performance
% The idea is using the sobol set to construct a trustable surrogate model at the beginning
fprintf('Warm up begins \n'); tic; % count warm up time
[Hk,Lambda,Nk,D,best_H]=warm_up_fcn(d,warm_up,left_bound,right_bound,fcn_name,noise_std);
fprintf('Warm up ends \n'); tWarmUp=toc; % count warm up time
fprintf('Warm up takes %8.4f seconds \n',tWarmUp);

%% MAIN LOOP
fprintf('Main loop begins.\n');
tic; % count main loop time
k=warm_up; % iteration counter
num_evaluation=warm_up; % budget consumption
while num_evaluation+1<=budget
    %% PROGRESS REPORT
    if mod(k,100)==0
        fprintf('iter: %5d, eval: %5d, cur best: %8.4f, true optimum: %8.4f \n',...
            k,num_evaluation,best_H(k),optimal_objective_value);
    end
    k=k+1;
    
    %% ADAPTIVE HYPERPARAMETERS
    [alpha,rk,t,numNumInt]=adaptive_hyper_para(k,warm_up,ca,gamma0,cr,gamma2,best_H(end));
    
    %% SAMPLING
    % given the sampling parameter theta_old=(mu_old,var_old)
    % generate a sample x from the independent multivariate normal density
    x_sample=normt_rnd(mu_old,var_old,left_bound,right_bound);
    Lambda(:,k)=x_sample; Nk(k)=1;
    
    %% FUNCTION EVALUATION
    cur_H=feval(fcn_name,x_sample); % current true objective function value 
    cur_h=cur_H+abs(cur_H)*normrnd(0,noise_std); % noisy observation
    num_evaluation=num_evaluation+1;
    best_H(k)=max(best_H(k-1),cur_H);
    
    %% PERFORMANCE ESTIMATIONS (asynchronous shrinking ball method)
    % given the current noisy observation cur_h
    % update the performance estimation Hk via shrinking ball method
    [Hk,Nk]=performance_estimation(Hk,Nk,cur_h,Lambda,k,rk,gamma1);
    
    %% SURROGATE MODELING
    % given Hk, construct the new surrogate model
    % based on a cubic model: S_k(x)=\sum_{i=1}^{k} weight(i)*||x-xi||^3
    [weight,D]=surrogate_model(Hk,D,Lambda,k);
    
    %% SAMPLING PARAMETER UPDATING
    % given the surrogate model (weight)
    % update the sampling parameter mu, var
    [mu_new,var_new,eta_x_new,eta_x2_new]=sampling_updating(mu_old,var_old,...
        left_bound,right_bound,eta_x_old,eta_x2_old,Lambda,weight,numNumInt,t,alpha);

    %% UPDATING
    eta_x_old=eta_x_new; eta_x2_old=eta_x2_new;
    mu_old=mu_new;  var_old = var_new;
    
    %% VISUALIZATION
    if mod(k,10)==0
        plot(best_H,'r-o'); % current best
        hold on
        cur_size=size(best_H);
        optimal_line=optimal_objective_value*ones(cur_size(2),1);
        plot(optimal_line,'k:','LineWidth',5); % true optimal value
        xlabel('Number of function evaluations')
        ylabel('Objective function value')
        title(sprintf('<%s>   Iteration: %5d  Evaluation: %5d',fcn_name,k,num_evaluation));
        legend('SMART','True optimal value','Location','southeast');
        ylim([best_H(1)*1.1 -0.8]);
        grid on
        drawnow;
    end
end

%% FINAL REPORT
fprintf('iter: %5d, eval: %5d, cur best: %8.4f, true optimum: %8.4f \n',...
    k,num_evaluation,best_H(k),optimal_objective_value);
fprintf('Main loop ends \n');
tMainLoop=toc; % count main loop time
fprintf('Main loop takes %8.4f seconds \n',tMainLoop);
