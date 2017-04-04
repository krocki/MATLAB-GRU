%
% gru_forward.m
%
% gru forward pass for grad check
%
% Author: Kamil Rocki <kmrocki@us.ibm.com>
% Created on: 02/04/2016
%

function loss = gru_forward(xs, target, Uz, Wz, bz, ...
                Ur, Wr, br, ...
                Uu, Wu, bu, Why, by, ...
                seq_length, h_prev)

	hidden_size = size(Wz, 1);
	vocab_size = size(Wz, 2);

	h = zeros(hidden_size, seq_length);
	y = zeros(vocab_size, seq_length);
	probs = zeros(vocab_size, seq_length);

	h(:, 1) = h_prev;

	loss = 0;

	for t = 2:seq_length

		%GRU gates
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
		loss = loss + sum(- log(probs(:, t)) .* target(:, t));

	end

end

