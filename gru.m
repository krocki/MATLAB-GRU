%
% gru.m
%
% GRU code
% based on http://arxiv.org/pdf/1412.3555v1.pdf
% and https://github.com/hassyGo/N3LP/blob/master/GRU.cpp
%
% Author: Kamil Rocki <kmrocki@us.ibm.com>
% Created on: 02/08/2016
%
% not optimized for shortness or performance
%

%read raw byte stream
data = read_raw('alice29.txt');

text_length = size(data, 1);

% alphabet
symbols = unique(data);
alphabet_size = size(symbols, 1);
ASCII_SIZE = 256;

% n_in - size of the alphabet, ex. 4 (ACGT)
n_in = ASCII_SIZE;
% n_out = n_in - predictions have the same size as inputs
n_out = n_in;

codes = eye(n_in);

max_iterations = text_length;
max_epochs = 1000;

observations = zeros(1, text_length);
perc = round(text_length / 100);
show_every = 10; %show stats every show_every s

% hyperparameters
hidden_size = 256; % size of hidden layer of neurons
seq_length = 50; % number of steps to unroll the RNN for
learning_rate = 1e-1;
vocab_size = n_in;

% model parameters
Wz = randn(hidden_size, vocab_size) * 0.01; % x to update gate
Uz = randn(hidden_size, hidden_size) * 0.01; % h_prev to update gate
bz = zeros(hidden_size, 1); %input gate bias

Wr = randn(hidden_size, vocab_size) * 0.01; % x to reset gate
Ur = randn(hidden_size, hidden_size) * 0.01; % h_prev to reset gate
br = zeros(hidden_size, 1); %output gate bias

Wu = randn(hidden_size, vocab_size) * 0.01; % x to candidate gate
Uu = randn(hidden_size, hidden_size) * 0.01; % h_prev to candidate gate
bu = zeros(hidden_size, 1); %input gate bias

Why = randn(vocab_size, hidden_size) * 0.01; % hidden to output
by = zeros(vocab_size, 1); % output bias

h = zeros(hidden_size, seq_length); % hidden
u = zeros(hidden_size, seq_length); % candidates
z = zeros(hidden_size, seq_length); % update gates
r = zeros(hidden_size, seq_length); % reset gates
rh = zeros(hidden_size, seq_length); % r(t) .* h(t-1)

%adagrad memory
mWhy = zeros(size(Why));
mby = zeros(size(by));

mWz = zeros(size(Wz));
mUz = zeros(size(Uz));
mbz = zeros(size(bz));

mWr = zeros(size(Wr));
mUr = zeros(size(Ur));
mbr = zeros(size(br));

mWu = zeros(size(Wu));
mUu = zeros(size(Uu));
mbu = zeros(size(bu));

target = zeros(vocab_size, seq_length);
y = zeros(vocab_size, seq_length);
dy = zeros(vocab_size, seq_length);
probs = zeros(vocab_size, seq_length);

%using log2 (bits), initial guess
smooth_loss = - log2(1.0 / alphabet_size);
loss_history = [];

%reset timer
tic

for e = 1:max_epochs
    
    
    %set some random context
    h(:, 1) = tanh(randn(size(h(:, 1))));

    %or zeros
    %h(:, 1) = zeros(size(h(:, 1)));
    
    beginning = randi([2 1+seq_length]); %randomize starting point
    for ii = beginning:seq_length:max_iterations - seq_length
        
        % reset grads
        dby = zeros(size(by));
        dWhy = zeros(size(Why));
        dy = zeros(size(target));
        dr = zeros(size(r));
        dz = zeros(size(z));
        du = zeros(size(u));
        drh = zeros(size(rh));

        dUz = zeros(size(Uz));
        dUr = zeros(size(Ur));
        dUu = zeros(size(Uu));

        dWz = zeros(size(Wz));
        dWr = zeros(size(Wr));
        dWu = zeros(size(Wu));

        dbz = zeros(size(bz));
        dbr = zeros(size(br));
        dbu = zeros(size(bu));

        dhnext = zeros(size(h(:, 1)));
        
        % get next symbol
        xs(:, 1:seq_length) = codes(data(ii - 1:ii + seq_length - 2), :)';
        target(:, 1:seq_length) = codes(data(ii:ii + seq_length - 1), :)';
        
        observations = char(data(ii - 1:ii + seq_length - 2))';
        t_observations = char(data(ii:ii + seq_length - 1))';
        
        % forward pass:
        
        loss = 0;
        
        for t = 2:seq_length
            
            % GRU gates
            % TODO: z and r can be computed at the same time with a single MMU
            z(:, t) = sigmoid(Wz * xs(:, t) + Uz * h(:, t - 1) + bz); % update gates
            r(:, t) = sigmoid(Wr * xs(:, t) + Ur * h(:, t - 1) + br); % reset gates
            
            % r(t) .* h(t-1)
            rh(:, t) = r(:, t) .* h(:, t - 1);

            % candidate value for hidden state
            u(:, t) = tanh(Wu * xs(:, t) + Uu * rh(:, t) + bu);
            
            % new hidden state
            h(:, t) = (1.0 - z(:, t)) .* h(:, t - 1) + z(:, t) .* u(:, t);
            
            % update y
            y(:, t) = Why * h(:, t) + by;
            
            % compute probs
            probs(:, t) = exp(y(:, t)) ./ sum(exp(y(:, t)));
            
            % cross-entropy loss, sum logs of probabilities of target outputs
            loss = loss + sum( -log2(probs(:, t)) .* target(:, t));
            
        end
        
        %bits/symbol
        loss = loss/seq_length;
        
        % backward pass:
        for t = seq_length: - 1:2
            
            % dy (global error)
            dy(:, t) = probs(:, t) - target(:, t); %dy[targets[t]] -= 1 # backprop into y
            dWhy = dWhy + dy(:, t) * h(:, t)'; %dWhy += np.doutt(dy, hs[t].T)
            dby = dby + dy(:, t); % dby += dy
            dh = Why' * dy(:, t) + dhnext; %dh = np.dot(Why.T, dy) + dhnext
            
            %dz = dh * (u(t) - h(t-1)) * sigmoid'(z(t))
            dz = dh .* (u(:, t) - h(:, t - 1)) .* (z(:, t) .* (1.0 - z(:, t)));
            %du = dh * (1 - z(t)) * tanh'(u(t))
            du = dh .* (1.0 - u(:, t) .* u(:, t)) .* z(:, t);
            %drh = Uu' * du
            drh = Uu' * du;
            %dr = drh * h(t-1) * sigmoid'(r(t))
            dr = drh .* h(:, t - 1) .* (r(:, t) .* (1.0 - r(:, t)));

            %linear layers
            dUz = dUz + dz * h(:, t - 1)';
            dWz = dWz + dz * xs(:, t)';
            dbz = dbz + dz;

            dUr = dUr + dr * h(:, t - 1)';
            dWr = dWr + dr * xs(:, t)';
            dbr = dbr + dr;

            dUu = dUu + du * rh(:, t)';
            dWu = dWu + du * xs(:, t)';
            dbu = dbu + du;

            dhnext = Ur' * dr + Uz' * dz + drh .* r(:, t) + dh .* (1.0 - z(:, t));

        end
        
        elapsed = toc;
        
        % debug code, checks gradients - slow!
        % if (elapsed > show_every)
        %     gru_grad_check;
        % end
        
        % clip gradients to some range

        dWhy = clip(dWhy, -5, 5);
        dby = clip(dby, -5, 5);
        
        dUz = clip(dUz, -5, 5);
        dWz = clip(dWz, -5, 5);
        dbz = clip(dbz, -5, 5);

        dUr = clip(dUr, -5, 5);
        dWr = clip(dWr, -5, 5);
        dbr = clip(dbr, -5, 5);

        dUu = clip(dUu, -5, 5);
        dWu = clip(dWu, -5, 5);
        dbu = clip(dbu, -5, 5);
        
        % % adjust weights, adagrad:
        mWhy = mWhy + dWhy .* dWhy;
        mby = mby + dby .* dby;
        
        mUz = mUz + dUz .* dUz;
        mWz = mWz + dWz .* dWz;
        mbz = mbz + dbz .* dbz;
        
        mUr = mUr + dUr .* dUr;
        mWr = mWr + dWr .* dWr;
        mbr = mbr + dbr .* dbr;
        
        mUu = mUu + dUu .* dUu;
        mWu = mWu + dWu .* dWu;
        mbu = mbu + dbu .* dbu;
        
        Why = Why - learning_rate * dWhy ./ (sqrt(mWhy + eps));
        by = by - learning_rate * dby ./ (sqrt(mby + eps));
        
        Uz = Uz - learning_rate * dUz ./ (sqrt(mUz + eps));
        Wz = Wz - learning_rate * dWz ./ (sqrt(mWz + eps));
        bz = bz - learning_rate * dbz ./ (sqrt(mbz + eps));
        
        Ur = Ur - learning_rate * dUr ./ (sqrt(mUr + eps));
        Wr = Wr - learning_rate * dWr ./ (sqrt(mWr + eps));
        br = br - learning_rate * dbr ./ (sqrt(mbr + eps));
        
        Uu = Uu - learning_rate * dUu ./ (sqrt(mUu + eps));
        Wu = Wu - learning_rate * dWu ./ (sqrt(mWu + eps));
        bu = bu - learning_rate * dbu ./ (sqrt(mbu + eps));
       
        % %%%%%%%%%%%%%%%%%%%%%
        
        smooth_loss = smooth_loss * 0.999 + loss * 0.001;
        
        % show stats every show_every s
        if (elapsed > show_every)
            
            loss_history = [loss_history smooth_loss];
            
            fprintf('[epoch %d] %d %% text read... smooth loss = %.3f\n', e, round(100 * ii / text_length), smooth_loss);
            fprintf('\n\nGenerating some text...\n');
            
            % random h,c seeds
            t = generate_gru(	Uz, Wz, bz, ...
                Ur, Wr, br, ...
                Uu, Wu, bu, ...
                Why, by, ...
                500, tanh(randn(size(h(:, 1)))));
            %t = generate_rnn(Wxh, Whh, Why, bh, by, 1000, clip(randn(size(Why, 2), 1) * 0.5, -1, 1));
            %generate according to the last seen h
            % t = generate_gru(	Uz, Wz, bz, ...
            %     Ur, Wr, br, ...
            %     Uu, Wu, bu, ...
            %     Why, by, ...
            %     500, h(:, seq_length));
            
            fprintf('%s \n', t);
            
            % update plots
            figure(1)
            
            subplot(6, 6, 1);
            imagesc(z + 0.5);
            title('z gates');
            subplot(6, 6, 7);
            imagesc(Wz');
            title('Wz');
            subplot(6, 6, 13);
            imagesc(Uz);
            title('Uz');
            
            subplot(6, 6, 2);
            imagesc(r + 0.5);
            title('r gates');
            subplot(6, 6, 8);
            imagesc(Wr');
            title('Wr');
            
            subplot(6, 6, 19);
            imagesc(dUz);
            title('dUz');
            
            subplot(6, 6, 14);
            imagesc(Ur);
            title('Ur');
            
            subplot(6, 6, 20);
            imagesc(dUr);
            title('dUr');
            
            subplot(6, 6, 21);
            imagesc(dUu);
            title('dUu');
            
            subplot(6, 6, 25);
            imagesc(dWz');
            title('dWz');
            
            subplot(6, 6, 26);
            imagesc(dWr');
            title('dWr');
            
            subplot(6, 6, 27);
            imagesc(dWu');
            title('dWu');
                    
            subplot(6, 6, 3);
            imagesc(rh + 0.5);
            title('rh');
            
            subplot(6, 6, 4);
            imagesc(u + 0.5);
            title('u');
            subplot(6, 6, 10);
            imagesc(Wu');
            title('Wu');
            subplot(6, 6, 16);
            imagesc(Uu);
            title('Uu');
                      
            subplot(6, 6, 6);
            imagesc((h + 1) / 2);
            title('h');
            
            subplot(6, 6, 11);
            imagesc(Why);
            title('Why');
            
            subplot(6, 6, 12);
            imagesc(dWhy);
            title('dWhy');
            
            subplot(6, 6, 18);
            plot(loss_history);
            title('Loss history');
            
            subplot(6, 6, 31);
            imagesc(xs);
            str = sprintf('xs: [%s]', observations);
            title(str);
            
            subplot(6, 6, 32);
            imagesc(target);
            str = sprintf('targets: [%s]', t_observations);
            title(str);
            
            subplot(6, 6, 33);
            imagesc(probs);
            title('probs');
            
            subplot(6, 6, 34);
            imagesc(dy);
            title('dy');
            
            drawnow;
            
            % reset timer
            tic
            
        end
        
        %carry
        z(:, 1) = z(:, seq_length);
        r(:, 1) = r(:, seq_length);
        u(:, 1) = u(:, seq_length);
        rh(:, 1) = rh(:, seq_length);
        h(:, 1) = h(:, seq_length);
        y(:, 1) = y(:, seq_length);
        probs(:, 1) = probs(:, seq_length);
        
    end
    
end