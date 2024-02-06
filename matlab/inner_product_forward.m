function [output] = inner_product_forward(input, layer, param)

d = size(input.data, 1);
k = size(input.data, 2); % batch size
n = size(param.w, 2);

% Replace the following line with your implementation.

output.height= input.height;
output.width = input.width;
output.channel = input.channel;
output.batch_size = k;
output.channel = input.channel;

output.data = zeros([n, k]);

%taking first input image from the batch and applying wx+b to get first
%output image of resulting output batch.
for i = 1:k
    output.data(:,i) = (param.w.' * input.data(:,i) ) + param.b.';
end

end
