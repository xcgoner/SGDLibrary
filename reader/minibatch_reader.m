function Reader = minibatch_reader(filepath, fileprefix, parser, n) 

    Reader.filelist = ls([filepath, fileprefix, '*']);
    
    Reader.getTest = @getTest;
    function [batch, reader] = getTest(m)
        X = cell(m, 1);
        y = cell(m, 1);
        idx = randperm(size(Reader.filelist, 1), m);
        parfor i = 1:m
            fullpath = [filepath, Reader.filelist(idx(i), :)];
            [X{i}, y{i}] = parser(fullpath);
        end
        batch.X = cell2mat(X) / 200;
        batch.X = [batch.X, ones(size(batch.X, 1), 1)];
        batch.y = cell2mat(y);
        % delete the test data from training data
        Reader.filelist(idx, :) = [];
        reader = Reader;
    end
    
    Reader.nextBatch = @nextBatch;
    function batch = nextBatch()
        X = cell(n, 1);
        y = cell(n, 1);
        idx = randperm(size(Reader.filelist, 1), n);
        parfor i = 1:n
            fullpath = [filepath, Reader.filelist(idx(i), :)];
            [X{i}, y{i}] = parser(fullpath);
        end
        batch.X = cell2mat(X) / 200;
        batch.X = [batch.X, ones(size(batch.X, 1), 1)];
        batch.y = cell2mat(y);
        
    end

end