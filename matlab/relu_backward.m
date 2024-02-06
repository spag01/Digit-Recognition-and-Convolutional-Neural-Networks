function [input_od] = relu_backward(output, input, layer)

% Replace the following line with your implementation.
data = input.data;
diff = output.diff;

input_od = zeros(size(data));


input_od = data > 0;

input_od = diff.*input_od;

end  
