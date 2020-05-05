#pragma once

#include <ATen/ATen.h>

void block_extractor_kernel_forward(
    at::Tensor& source_a,
    at::Tensor& source_b,
    at::Tensor& source_c,
    at::Tensor& flow_field_a, 
    at::Tensor& flow_field_b, 
    at::Tensor& flow_field_c, 
    at::Tensor& masks_a,
    at::Tensor& masks_b,
    at::Tensor& masks_c,
    at::Tensor& output,
    int kernel_size);

void block_extractor_kernel_backward(
    at::Tensor& source_a,
    at::Tensor& source_b, 
    at::Tensor& source_c, 
    at::Tensor& flow_field_a,
    at::Tensor& flow_field_b,
    at::Tensor& flow_field_c,
    at::Tensor& masks_a,
    at::Tensor& masks_b,
    at::Tensor& masks_c,
    at::Tensor& grad_output,
    at::Tensor& grad_source_a, 
    at::Tensor& grad_source_b, 
    at::Tensor& grad_source_c, 
    at::Tensor& grad_flow_field_a, 
    at::Tensor& grad_flow_field_b,
    at::Tensor& grad_flow_field_c,
    int kernel_size);