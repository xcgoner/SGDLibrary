% reader = minibatch_reader('../../../data_new/CIKM2017_train/train_dir/', 'cikm_train_', @cikm2017_rainfall_parser, 100);
% [test_data, reader] = reader.getTest(10);


% [X, y] = cikm2017_rainfall_parser('../../../data_new/CIKM2017_train/train_dir/cikm_train_ab');

[X, y, idx] = cikm2017_rainfall_index_parser('../../../data_new/CIKM2017_testA/test_dir/cikm_test_0000');
