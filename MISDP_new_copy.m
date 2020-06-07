% This script contains the complete formulation of the MISDP problem.
% minimize 1/A_n \sum_{i =1}^{A_n} tr(P_i)
% delta_ is block diagonal matrix containing \Delta_1, \Delta_2, ...,
% \Delta_n. 
% delta is the block column matrix containing \Delta_1, \Delta_2, ...,
% \Delta_n. 

% clc
% clearvars
% close all
% 
% Read CSV of current adjacency matrix
% adj_mat = readmatrix('simulation_data/adj_7_3_5_100_1.csv');
% adj_mat = readmatrix('simulation_data/adj_mat.csv');
% 
% % Read CSV of current covariance matrices
% cov_dir = './simulation_data/inverse_covariance_matrices/';
% cov_matrices = dir(fullfile(cov_dir, '*.csv'));
% Om_ = [];
% for cov=cov_matrices'
%     o = readmatrix(fullfile(cov_dir, cov.name));
%     Om_ = [Om_; o];
% end
% 
% ne=3;

% do_misdp(adj_mat, adj_mat, Om_, size(o), 3);
% do_misdp(3);

% This script contains the complete formulation of the MISDP problem.
% minimize 1/A_n \sum_{i =1}^{A_n} tr(P_i)
% delta_ is block diagonal matrix containing \Delta_1, \Delta_2, ...,
% \Delta_n. 
% delta is the block column matrix containing \Delta_1, \Delta_2, ...,
% \Delta_n. 

% function [adj_mat, weights] = do_misdp(currA, currW, Om_, Om_size, ne)
function [A, PI] = MISDP_new_copy(ne)
    addpath(genpath('/Users/nicole/USC/RESL_Summer_Project/mtt/YALMIP-master'));
    % Read CSV of current adjacency matrix
    adj_mat = readmatrix('misdp_data/adj_mat.csv');

    % Read CSV of current omega matrices
    % disp('reducing vals for computation purposes');
    omega_dir = './misdp_data/omega_matrices/';
    omega_matrices = dir(fullfile(omega_dir, '*.csv'));
    Om_ = [];
    for omega=omega_matrices'
        o = readmatrix(fullfile(omega_dir, omega.name)) * 0.01;
        Om_ = [Om_; o];
    end

    % Read CSV of current covariance matrices
    % disp('reducing vals for computation purposes');
    cov_dir = './misdp_data/inverse_covariance_matrices/';
    cov_matrices = dir(fullfile(cov_dir, '*.csv'));
    Um_ = [];
    for cov=cov_matrices'
        u = readmatrix(fullfile(cov_dir, cov.name)) * 0.01;
        Um_ = [Um_; u];
    end
    
    Om_size = size(o);
    currA = adj_mat;
    size_A = size(currA);
    A_n = size_A(1);
    s = Om_size(1);
    n = A_n;
    Pi = double(currA);
    delta_c = cell(A_n,1);

    thres = 10;
    A = sdpvar(n);

    M_ = [];
    for i = 1:A_n
        M_= blkdiag(M_, sdpvar(s));
    end
    delta_ = [];
    delta = [];
    for i = 1:A_n
        delta_c{i} = sdpvar(s);
        delta = [delta; delta_c{i}];
        delta_= blkdiag(delta_, delta_c{i});
    end

    beta = 1/n;
    mu = sdpvar(1);
    d = ones(n,1)*10;
    PI = binvar(n,n);
    obj =  1/A_n * trace(M_);
    F = [];


    % Scur complement based covariance is inverse of information matrix
    schur = sdpvar([M_ eye(size(delta_));
        eye(size(M_)) delta_]);

    F = [F, schur >= 0];

    %F = [F, kron(A, eye(s)) * Om_ == delta];
    F = [F, kron(A, eye(s)) * Om_ + Um_ == delta];


    F = [F, A*ones(n,1) == ones(n,1)];
    F = [F, beta*ones(n) + (1-mu)*eye(n)  >= A, mu >= 0.00001];
%     F = [F, norm(PI-Pi, 'fro')^2 <= ne];
    F = [F, norm(PI-Pi, 'fro')^2 <= 2*ne];

    for i = 1:n
        for j = 1:n
            if i == j
                F = [F, A(i,j) >=0.00001, PI(i,j) == 1];
            else
            end
            F = [F, A(i,j) >= 0];
            F = [F, A(i,j) <= 1 * PI(i,j)];
        end
    end

    options = sdpsettings('verbose',0,...
    'solver','bnb','bnb.solver','mosek','bnb.maxiter',1000, 'debug', 1);    

    try
        % solve the problem
        disp('solving');
        sol = optimize(F,obj,options);

        % Analyze the solution
        if sol.problem == 0
            disp('solution found');
            A=value(A);
            PI=value(PI);
            writematrix(A, 'misdp_data/new_weights.csv');
            writematrix(PI, 'misdp_data/new_A.csv');
        else
            sol.info
            yalmiperror(sol.problem);
            r_P = [];
            disp('no solution found');
            % python will read old weights
            writematrix(adj_mat, 'misdp_data/new_A.csv');
            fileID = fopen('fail.txt','w');
            fprintf(fileID,'fail try once more');
            fclose(fileID);
        end
    catch
        disp('branch and bound problem');
        % python will read old weights
        writematrix(adj_mat, 'misdp_data/new_A.csv');
        fileID = fopen('misdp_data/fail.txt','w');
        fprintf(fileID,'fail try once more');
        fclose(fileID);
    
%    for cov=cov_matrices'
%        delete(fullfile(cov_dir, cov.name));
%     end
        
end 