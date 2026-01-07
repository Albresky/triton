import torch
import triton
import triton.language as tl
DEVICE = triton.runtime.driver.active.get_active_torch_device(3)

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, ):
    pid = tl.program_id(axis=0) 
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.device_print("tl.device_print: ", x)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=64, num_warps=2)
    return output

torch.manual_seed(0)
size = 64
x = torch.arange(0,size,device=DEVICE)
y = torch.arange(0,size,device=DEVICE)
print(f'\n\nbuild_in print(x):{x}')
output_torch = x + y
output_triton = add(x, y)

print('\n\noutput:')
print(f'output_torch: {output_torch}')
print(f'output_triton: {output_triton}')
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

