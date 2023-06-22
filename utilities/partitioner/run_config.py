# NOTE: this reads in feature matrix (and connections matrix for joint nmf), partitions it for
# PLANC, runs PLANC, then prints out final error to compare with the error computed by PLANC
# (currently can't do this for joint-nmf because we don;t save h_hat). Modify
# included config file to change the details of the run.


# USAGE: python3 test_error.py path_to_config_file config_option
# python3 test_error.py ./configs/test.json run_nmf_cit
# python3 test_error.py ./configs/test.json run_joint_nmf_cit | grep err
# python3 test_error.py ./configs/test.json run_joint_nmf_cit | grep -i sqnormA

from cgi import test
from scipy.io import mmread
import subprocess as sb         #for running PLANC
import sys
import json
from sklearn import decomposition
import numpy as np
from scipy.linalg import norm
from scipy import sparse
from planc_partitioner import planc_partitioner
import os

config_path = sys.argv[1]
f = open(config_path)
config = json.load(f)
f.close()

id = sys.argv[2]
c = config[id][0]

print("config type: " + str(type(config)))
print(list(config.keys()))

print(id)
print(c)

print(c['format'])

# read in dataset ----> mm stands for matrix market format...
if c['format'] == "mm" :
    A = None
    m = 0
    n = 0
    k = c['k']

    

    # partition feature matrix... ----> writing new partition function
    test = None

    planc_partitioned_feat_mat_path = None
    planc_partitioned_conn_mat_path = None

    if c['partition']:
        print("reading feature matrix in matrix market format")
        A = mmread(c['feature_mat_path'])
        m = A.shape[0]
        n = A.shape[1]
        
        print(type(A))
        print(A)
        print("c['partition'] set to TRUE, partitioning feature matrix")
        test = planc_partitioner() 

        planc_partitioned_feat_mat_path = test.partition(c['feature_mat_path'], c['partitioner_output_path'] if ('partitioner_output_path' in c) else os.path.dirname(c['feature_mat_path']), A,m,n,c['pr'], c['pc'])
    else:
        print("c['partition'] set to FALSE, not partitioning feature matrix")
        m = c['m']
        n = c['n']

        planc_partitioned_feat_mat_path = os.path.join(c['partitioner_output_path'], os.path.basename(c['feature_mat_path'])) if ('partitioner_output_path' in c) else os.path.abspath(c['feature_mat_path'])

    print("planc_partitioned_feat_mat_path: " + planc_partitioned_feat_mat_path)

    output_path = '{}_{}_{}_{}'.format(c['output_path'], str(c['iterations']), str(c['algorithm']), str(k))

    # run PLANC on A... ----> NOTE: alpha and beta not used for now...
    if c['algorithm'] == 9:

        if c['partition']:
            S = mmread(c['conn_mat_path'])
            assert n == S.shape[0] == S.shape[1]
            print("c['partition'] set to TRUE, partitioning connection matrix")
            planc_partitioned_conn_mat_path = test.partition(c['conn_mat_path'], c['partitioner_output_path'] if ('partitioner_output_path' in c) else os.path.dirname(c['conn_mat_path']), S, n,n,c['pr'], c['pc'])
        else:
            print("c['partition'] set to FALSE, not partitioning connection matrix")
            planc_partitioned_conn_mat_path = os.path.join(c['partitioner_output_path'], os.path.basename(c['feature_mat_path'])) if ('partitioner_output_path' in c) else os.path.abspath(c['conn_mat_path'])

        print("planc_partitioned_conn_mat_path: " + planc_partitioned_conn_mat_path)

        planc_command = '{} -np {} {} -a {} -k {} -i {} -c {} -d \"{} {}\"  -t {} -p \"{} {}\" -o {} -e {} {}'.format(c['mpirun_path'], c['procs'], c['planc_path'], c['algorithm'], k, planc_partitioned_feat_mat_path, planc_partitioned_conn_mat_path, m, n, c['iterations'], c['pr'], c['pc'], output_path, c['error'], c['debug'])
    else:
        planc_command = '{} -np {} {} -a {} -k {} -i {} -d \"{} {}\"  -t {} -p \"{} {}\" -o {} -e {} {}'.format(c['mpirun_path'], c['procs'], c['planc_path'], c['algorithm'], k, planc_partitioned_feat_mat_path, m, n, c['iterations'], c['pr'], c['pc'], output_path, c['error'], c['debug'])
    
    print(planc_command)

    sb.run(planc_command, shell=True)

    # read in output W and H (using either python or matlab.engine), compute error using
    # matlab, python, and planc, then check to see if the error computed by planc matches that of
    # the other two...

    if c['test_error']:
        print("c['test_error'] set to TRUE")


        W_path = output_path + '_W'
        H_path = output_path + '_H'

    # NOTE: is float64 the correct data-type?...
        W = np.fromfile(W_path, 'float64')
        W = np.asmatrix(W).reshape(k, m)


        H = np.fromfile(H_path, 'float64')
        H = np.asmatrix(H).reshape(k,n)

    # use Python to compute expected error...
        if c['algorithm'] == 9:
            print("err: NOTE NEED TO SAVE H_HAT TO COMPUTE JOINT-NMF ERROR")
        else:
            print("err: " + str(norm(A - W.transpose()*H, 'fro')))

        print("attempting to check error with MatLab engine. This is currently untested and may only work for the cit-hepTH dataset")
        try:
            # NOTE: this part is untested...
            import matlab.engine
            # TODO: recreate topic over time results using PLANC...
            # deploy the matlab engine..,.
            eng = matlab.engine.start_matlab()

            eng.workspace['H_path'] = H_path
            eng.workspace['W_path'] = W_path
            eng.workspace['metadata_path'] = c['metadata_path']
            eng.workspace['doc_counts'] = n
            eng.workspace['term_counts'] = m
            eng.workspace['k'] = k
            eng.workspace['matlab_output_path'] = c['matlab_output_path']
            eng.workspace['alg'] = c['algorithm']
            eng.eval("""

            who

            addpath(\'./matlab/\')

            f = fopen(W_path, 'r');
            test_w = fread(f,[980, k], 'float64');
            size(test_w)

            f = fopen(H_path, 'r');
            test_h = fread(f,[k, doc_counts], 'float64');
            size(test_h)

            % normalize columns of w and h...
            % test_w = normc(test_w);
            % test_h = normc(test_h);
            % normr(test_w)
            % normc(test_h)
            % test_w = test_w./sum(test_w);
            % test_h = test_h./sum(test_h);
            % sum(test_h)
            % test_w

            metadata = load(metadata_path);

            [c,ia,ic] = unique(metadata.years, 'stable');
            time_offset = accumarray(ic, 1)

            terms = metadata.vocab';

            topic_year_weights = zeros(k, length(time_offset));

            [Y, I] = sort(c)

            % Average the columns corresponding to each year together
            year_sum = 0
            for i = 1:length(time_offset)
                for j = 1:time_offset(i)
                    topic_year_weights(:,I(i)) = topic_year_weights(:,I(i)) + test_h(:, year_sum + j);
                end
                year_sum = year_sum + time_offset(i);
                topic_year_weights(:,i) = topic_year_weights(:,i)/time_offset(i);
            end

            topic_year_weights = topic_year_weights./sum(topic_year_weights);
            % topic_year_weights = normc(topic_year_weights);

            tw = 50;

            M = topic_words(test_w, tw);

            temp = cell(k,1);
            for i = 1:k
            temp(i,1) = mat2cell(i,1);
            end

            var_names = string(temp)

            topic_table = array2table(zeros(tw,k), 'VariableNames', var_names);

            for i = 1:k
                topic_table.(int2str(i)) = terms(M(:,i), :);
            end

            t = linspace(1992,2003,12);


            % set(gca, 'ColorOrder', jet(k));

            figure

            subplot(1,2,1);

            %topic_year_weights_smooth = topic_year_weights
            for i = 1:k
                topic_year_weights(i, :) = smooth(topic_year_weights(i, :));
            end

            topic_year_weights'

            plt = plot(t,topic_year_weights', 'LineWidth', 4)

            hold all
            set(gca, 'ColorOrder', distinguishable_colors(k));
            legend(var_names)
            % legend([plt], var_names) %'1', '2', '3', '4', '5')
            % legend

            xlabel("Year")
            ylabel("Smoothed Average Document Topic Factor Weight")


            hax = subplot(1,2,2);

            hui = uitable('Data', topic_table{:,:}, ...
                'ColumnName', topic_table.Properties.VariableNames, ...
                'RowName', topic_table.Properties.RowNames, ...
                'Units','normalized','Position', hax.Position);

            delete(hax)

            %matlab_output_path = '/Users/bencobb/Desktop/Projects/PLANC/matlab_scratch/topic_time/output/' 

            fname = strcat(matlab_output_path,'plot_planc_joint_nmf','_',string(datestr(now,'hh_MM_ss_mm_dd_yy')),'_',string(alg),'_',string(k),'.fig')

            % title('alpha = default, beta = default')

            savefig(fname)

            % time_offset = 
            % print(metadata)

            who

            """, nargout = 0)

            print('c[\'matlab_output_path\']: ' + c['matlab_output_path'])
            print('output_path: ' + output_path)

        except Exception:
            print("Couldn't \"import matlab.engine\", continuing on without it")
            traceback.print_exec()

    else:
        print("c['test_error'] set to FALSE")



# TODO: use sklearn to compute nmf and repeat above topics over time process...



# else:
#     print("format not supported")