function [ out ] = clip( x, r1, r2 )

    out = x;

    out = max(out, r1);
    out = min(out, r2);

end

