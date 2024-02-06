function [output] = pooling_layer_forward(input, layer)

    h_in = input.height;
    w_in = input.width;
    c = input.channel;
    batch_size = input.batch_size;
    k = layer.k;
    pad = layer.pad;
    stride = layer.stride;
    
    h_out = (h_in + 2*pad - k) / stride + 1;
    w_out = (w_in + 2*pad - k) / stride + 1;
    
    
    output.height = h_out;
    output.width = w_out;
    output.channel = c;
    output.batch_size = batch_size;

    % Replace the following line with your implementation.
    output.data = zeros([h_out * w_out * c, batch_size]);
    temp_data = zeros([h_out, w_out, c, batch_size]);

    reshape_data = reshape(input.data,h_in, w_in, c, batch_size);

    for i = 1 : batch_size
        for j = 1 : c
            for p = 1 : h_out
                for q = 1 : w_out 
                    filter = reshape_data((p-1) * stride+1 : (p-1) * stride+k, (q-1) * stride+1 : (q-1) * stride+k, j, i);
                    temp_data(p,q,j,i) = max(filter(:));
                end
            end
        end
    end
    
    output.data = reshape(temp_data, h_out * w_out * c, batch_size);

end

    h_in = input['height']
    w_in = input['width']
    c = input['channel']
    batch_size = input['batch_size']
    k = layer['k']
    pad = layer['pad']
    stride = layer['stride']

    h_out = int((h_in + 2 * pad - k) / stride + 1)
    w_out = int((w_in + 2 * pad - k) / stride + 1)
    
    output = {}
    output['height'] = h_out
    output['width'] = w_out
    output['channel'] = c
    output['batch_size'] = batch_size
    output['data'] = np.zeros((h_out, w_out, c, batch_size)) # replace with your implementation
