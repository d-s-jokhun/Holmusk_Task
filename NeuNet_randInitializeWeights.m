function W = NeuNet_randInitializeWeights(L_in, L_out)
% This function ramdomly initializes weights in a matrix of L_out rows and
% L_in columns

epsilon_init = 0.12; % range in which the initial weights will lie
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init; % Random number between -epsilon_init and epsilon_init to break the symmetry


end
