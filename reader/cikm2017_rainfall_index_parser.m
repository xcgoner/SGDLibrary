function [X, y, idx] = cikm2017_rainfall_index_parser(fullpath)

    dt = readtable(fullpath, 'Format', '%s%f%s', 'Delimiter', ',');
    
    n = size(dt, 1);
    
    y = dt.Var2;
    
    X = cell(1, n);
    
    for i = 1:n
        X(i) = textscan(dt.Var3{i}, '');
    end
    X = cell2mat(X)';
    
    idx = str2double(regexprep(dt.Var1, '[a-z]|_|[A-Z]', ''));
       
end