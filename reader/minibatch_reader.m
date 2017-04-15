function Reader = minibatch_reader(filepath, fileprefix, parser, n) 

    Reader.filelist = ls([filepath, fileprefix, '*']);
    
    Reader.nextBatch = @nextBatch;
    function batch = nextBatch()

        X = cell(n, 1);
        y = cell(n, 1);
        idx = randperm(size(Reader.filelist, 1), n);
        for i = 1:n
            fullpath = [filepath, Reader.filelist(idx(i), :)];
            [X{i}, y{i}] = parser(fullpath);
        end
        batch.X = cell2mat(X);
        batch.y = cell2mat(y);
        
    end

end