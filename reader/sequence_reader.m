function Reader = sequence_reader(filepath, fileprefix, parser) 

    Reader.filelist = ls([filepath, fileprefix, '*']);
    
    Reader.nextBatch = @nextBatch;
    function [batch, reader] = nextBatch()
        if isempty(Reader.filelist)
            batch = [];
        else
            fullpath = [filepath, Reader.filelist(1, :)];
            [X, y, idx] = parser(fullpath);
            batch.X = X / 200;
            batch.X = [batch.X, ones(size(batch.X, 1), 1)];
            batch.y = y;
            batch.idx = idx;
            Reader.filelist(1, :) = [];
        end
        
        reader = Reader;
        
    end

end