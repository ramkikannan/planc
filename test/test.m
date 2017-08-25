%A=randn(5000,4000);
%W_init=randn(5000, 20);
%H_init=randn(20,4000);
A=rand(50000,40000);
W_init = rand(50000,20);
H_init=rand(20,40000);
matinit.W=single(W_init);
matinit.H=single(H_init);
echo on;
norm(A,'fro')
norm(matinit.W,'fro')
norm(matinit.H,'fro')
[W_arma, H_arma, WtW_arma, AtW_arma, HtH_arma, AH_arma]=matlabvsarma(A,W_init,H_init',20);
addpath(genpath('/ccs/home/ramki/matlablibraries'));
[W_matlab, H_matlab]= nmf(A,20,'init',matinit,'max_iter',20,'method','anls_bpp');
norm(W_matlab-W_arma,'fro')
norm(H_matlab'-H_arma,'fro')
norm(A-W_matlab*H_matlab,'fro')
norm(A-W_arma*H_arma','fro')
norm(matinit.H*matinit.H'-HtH_arma, 'fro')
norm(matinit.W'*matinit.W-WtW_arma, 'fro')
norm(A'*matinit.W-AtW_arma, 'fro')
norm(A*matinit.H'-AH_arma, 'fro')
echo off


