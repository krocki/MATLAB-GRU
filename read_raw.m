function [data] = read_raw(filename)

    fid = fopen(filename, 'r');
    data = fread(fid);

end