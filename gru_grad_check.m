%
% gru_grad_check.m
%
% gradient check for LSTM code
%
% Author: Kamil Rocki <kmrocki@us.ibm.com>
% Created on: 02/04/2016
%
% didn't have time to make it more beautiful
%

dby_err = zeros(vocab_size, 1);
dWhy_err = zeros(vocab_size, hidden_size);
dUz_err = zeros(hidden_size, hidden_size);
dUr_err = zeros(hidden_size, hidden_size);
dUu_err = zeros(hidden_size, hidden_size);

dWz_err = zeros(hidden_size, vocab_size);
dWr_err = zeros(hidden_size, vocab_size);
dWu_err = zeros(hidden_size, vocab_size);

nby = zeros(vocab_size, 1);
nWhy = zeros(vocab_size, hidden_size);
nUz = zeros(hidden_size, hidden_size);
nUr = zeros(hidden_size, hidden_size);
nUu = zeros(hidden_size, hidden_size);
nWz = zeros(hidden_size, vocab_size);
nWr = zeros(hidden_size, vocab_size);
nWu = zeros(hidden_size, vocab_size);

increment = 1e-3;

for k=1:vocab_size
    delta = zeros(vocab_size, 1);
    delta(k) = increment;
    
    pre_loss = gru_forward(xs, target, Uz, Wz, bz, Ur, Wr, br, Uu, Wu, bu, Why, by - delta, seq_length, h(:, 1));
    post_loss = gru_forward(xs, target, Uz, Wz, bz, Ur, Wr, br, Uu, Wu, bu, Why, by + delta, seq_length, h(:, 1));
    
    numerical_grad = (post_loss - pre_loss) / (increment * 2);
    nby(k) = numerical_grad;
    analitic_grad = dby(k);
    dby_err(k) = abs(numerical_grad - analitic_grad)/abs(numerical_grad + analitic_grad);
    
end

%dWhy
for k=1:vocab_size
    for kk=1:hidden_size
        
        delta = zeros(vocab_size, hidden_size);
        delta(k, kk) = increment;
        
        pre_loss = gru_forward(xs, target, Uz, Wz, bz, Ur, Wr, br, Uu, Wu, bu, Why - delta, by, seq_length, h(:, 1));
        post_loss = gru_forward(xs, target, Uz, Wz, bz, Ur, Wr, br, Uu, Wu, bu, Why + delta, by, seq_length, h(:, 1));
        
        numerical_grad = (post_loss - pre_loss) / (increment * 2);
        nWhy(k, kk) = numerical_grad;
        analitic_grad = dWhy(k, kk);
        dWhy_err(k, kk) = abs(numerical_grad - analitic_grad)/abs(numerical_grad + analitic_grad);
        
    end
end

% dUz
for k=1:hidden_size
    for kk=1:hidden_size
        
        delta = zeros(hidden_size, hidden_size);
        delta(k, kk) = increment;
        
        pre_loss = gru_forward(xs, target, Uz - delta, Wz, bz, Ur, Wr, br, Uu, Wu, bu, Why, by, seq_length, h(:, 1));
        post_loss = gru_forward(xs, target, Uz + delta, Wz, bz, Ur, Wr, br, Uu, Wu, bu, Why, by, seq_length, h(:, 1));
        
        numerical_grad = (post_loss - pre_loss) / (increment * 2);
        nUz(k, kk) = numerical_grad;
        analitic_grad = dUz(k, kk);
        dUz_err(k, kk) = abs(numerical_grad - analitic_grad)/abs(numerical_grad + analitic_grad);
        
    end
end

%dUr
for k=1:hidden_size
    for kk=1:hidden_size
        
        delta = zeros(hidden_size, hidden_size);
        delta(k, kk) = increment;
        
        pre_loss = gru_forward(xs, target, Uz, Wz, bz, Ur - delta, Wr, br, Uu, Wu, bu, Why, by, seq_length, h(:, 1));
        post_loss = gru_forward(xs, target, Uz, Wz, bz, Ur + delta, Wr, br, Uu, Wu, bu, Why, by, seq_length, h(:, 1));
        
        numerical_grad = (post_loss - pre_loss) / (increment * 2);
        nUr(k, kk) = numerical_grad;
        analitic_grad = dUr(k, kk);
        dUr_err(k, kk) = abs(numerical_grad - analitic_grad)/abs(numerical_grad + analitic_grad);
        
    end
end

%dUu
for k=1:hidden_size
    for kk=1:hidden_size
        
        delta = zeros(hidden_size, hidden_size);
        delta(k, kk) = increment;
        
        pre_loss = gru_forward(xs, target, Uz, Wz, bz, Ur, Wr, br, Uu - delta, Wu, bu, Why, by, seq_length, h(:, 1));
        post_loss = gru_forward(xs, target, Uz, Wz, bz, Ur, Wr, br, Uu + delta, Wu, bu, Why, by, seq_length, h(:, 1));
        
        numerical_grad = (post_loss - pre_loss) / (increment * 2);
        nUu(k, kk) = numerical_grad;
        analitic_grad = dUu(k, kk);
        dUu_err(k, kk) = abs(numerical_grad - analitic_grad)/abs(numerical_grad + analitic_grad);
        
    end
end

%dWz
for k=1:hidden_size
    for kk=1:vocab_size
        
        delta = zeros(hidden_size, vocab_size);
        delta(k, kk) = increment;
        
        pre_loss = gru_forward(xs, target, Uz, Wz - delta, bz, Ur, Wr, br, Uu, Wu, bu, Why, by, seq_length, h(:, 1));
        post_loss = gru_forward(xs, target, Uz, Wz + delta, bz, Ur, Wr, br, Uu, Wu, bu, Why, by, seq_length, h(:, 1));
        
        numerical_grad = (post_loss - pre_loss) / (increment * 2);
        nWz(k, kk) = numerical_grad;
        analitic_grad = dWz(k, kk);
        dWz_err(k, kk) = abs(numerical_grad - analitic_grad)/abs(numerical_grad + analitic_grad);
        
    end
end

%dWr
for k=1:hidden_size
    for kk=1:vocab_size
        
        delta = zeros(hidden_size, vocab_size);
        delta(k, kk) = increment;
        
        pre_loss = gru_forward(xs, target, Uz, Wz, bz, Ur, Wr - delta, br, Uu, Wu, bu, Why, by, seq_length, h(:, 1));
        post_loss = gru_forward(xs, target, Uz, Wz, bz, Ur, Wr + delta, br, Uu, Wu, bu, Why, by, seq_length, h(:, 1));
        
        numerical_grad = (post_loss - pre_loss) / (increment * 2);
        nWr(k, kk) = numerical_grad;
        analitic_grad = dWr(k, kk);
        dWr_err(k, kk) = abs(numerical_grad - analitic_grad)/abs(numerical_grad + analitic_grad);
        
    end
end

%dWu
for k=1:hidden_size
    for kk=1:vocab_size
        
        delta = zeros(size(Wu));
        delta(k, kk) = increment;
        
        pre_loss = gru_forward(xs, target, Uz, Wz, bz, Ur, Wr, br, Uu, Wu - delta, bu, Why, by, seq_length, h(:, 1));
        post_loss = gru_forward(xs, target, Uz, Wz, bz, Ur, Wr, br, Uu, Wu + delta, bu, Why, by, seq_length, h(:, 1));
        
        numerical_grad = (post_loss - pre_loss) / (increment * 2);
        nWu(k, kk) = numerical_grad;
        analitic_grad = dWu(k, kk);
        dWu_err(k, kk) = abs(numerical_grad - analitic_grad)/abs(numerical_grad + analitic_grad);
        
    end
end

%all errors should be relatively small - 1e-6 or less
fprintf('dby err = %.12f, max dby = %.12f, max nby = %.12f\n', max(dby_err(:)), max(dby(:)), max(nby(:)));
fprintf('dWhy err = %.12f, max dWhy = %.12f, max nWhy = %.12f\n', max(dWhy_err(:)), max(dWhy(:)), max(nWhy(:)));
fprintf('dUz err = %.12f, max dUz = %.12f, max nUz = %.12f\n', max(dUz_err(:)), max(dUz(:)), max(nUz(:)));
fprintf('dUr err = %.12f, max dUr = %.12f, max nUr = %.12f\n', max(dUr_err(:)), max(dUr(:)), max(nUr(:)));
fprintf('dUu err = %.12f, max dUu = %.12f, max nUu = %.12f\n', max(dUu_err(:)), max(dUu(:)), max(nUu(:)));
fprintf('dWz err = %.12f, max dWz = %.12f, max nWz = %.12f\n', max(dWz_err(:)), max(dWz(:)), max(nWz(:)));
fprintf('dWr err = %.12f, max dWr = %.12f, max nWr = %.12f\n', max(dWr_err(:)), max(dWr(:)), max(nWr(:)));
fprintf('dWu err = %.12f, max dWu = %.12f, max nWu = %.12f\n', max(dWu_err(:)), max(dWu(:)), max(nWu(:)));

