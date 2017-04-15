test_reader = minibatch_reader('../../../data_new/CIKM2017_train/train_dir/', 'cikm_train_', @cikm2017_rainfall_parser, 100);
test_data = reader.nextBatch();

% [X, y] = cikm2017_rainfall_parser('../../../data_new/CIKM2017_train/train_dir/cikm_train_ab');
