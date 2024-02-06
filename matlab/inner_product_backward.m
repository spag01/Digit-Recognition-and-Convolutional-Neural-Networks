function [param_grad, input_od] = inner_product_backward(output, input, layer, param)

% Replace the following lines with your implementation.
data = input.data;
diff = output.diff;
weights = param.w;

param_grad.b = sum(diff, 2).';
param_grad.w = data * (diff.');
input_od = weights * diff;
end