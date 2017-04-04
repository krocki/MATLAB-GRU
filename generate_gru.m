%
% generate_gru.m
%
% LSTM code
% based on http://deeplearning.net/tutorial/gru.html
%
% Author: Kamil Rocki <kmrocki@us.ibm.com>
% Created on: 02/03/2016
%
% generate a sequence given a LSTM network
%

function text = generate_gru(Uz, Wz, bz, ...
                Ur, Wr, br, ...
                Uu, Wu, bu, Why, by, l, h)

    text = [];
    codes = eye(size(Why, 1));

    for i=1:l-1
        
        y = Why * h + by;
        probs = exp(y)./sum(exp(y));
        cdf = cumsum(probs);

        r = rand();
        sample = min(find(r <= cdf));
        text = [text char(sample)];

        %update hidden state
        x = codes(sample, :)';

        z =  sigmoid(Wz * x + Uz * h + bz);
        r =  sigmoid(Wr * x + Ur * h + br);
        rh = r .* h;
        u = tanh(Wu * x + Uu * rh + bu);

        h = (1.0 - z) .* h + z .* u;
        
    end

end