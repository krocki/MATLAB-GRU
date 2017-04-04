function [ out ] = sigmoid( x )

    out = 1./(1 + exp(-1*x));
    
end

