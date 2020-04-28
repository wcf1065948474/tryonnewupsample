from torch.nn.modules.module import Module
from torch.autograd import Function, Variable
import mutil_block_extractor_cuda

class MutilBlockExtractorFunction(Function):

    @staticmethod
    def forward(ctx, source_a, source_b, source_c, flow_field_a, flow_field_b, flow_field_c, mask_a, mask_b, mask_c, kernel_size):
        assert source_a.is_contiguous()
        assert source_b.is_contiguous()
        assert source_c.is_contiguous()
        assert flow_field_a.is_contiguous()
        assert flow_field_b.is_contiguous()
        assert flow_field_c.is_contiguous()
        assert mask_a.is_contiguous()
        assert mask_b.is_contiguous()
        assert mask_c.is_contiguous()

        # TODO: check the shape of the inputs 
        bs, ds, hs, ws = source_a.size()
        bf, df, hf, wf = flow_field_a.size()
        # assert bs==bf and hs==hf and ws==wf
        assert df==2

        ctx.save_for_backward(source_a, source_b, source_c, flow_field_a, flow_field_b, flow_field_c, mask_a, mask_b, mask_c)
        ctx.kernel_size = kernel_size

        output = flow_field_a.new(bs, ds, kernel_size*hf, kernel_size*wf).zero_()

        if not source_a.is_cuda:
            raise NotImplementedError
        else:
            mutil_block_extractor_cuda.forward(source_a, source_b, source_c, flow_field_a, flow_field_b, flow_field_c, mask_a, mask_b, mask_c, output, kernel_size)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output.contiguous()
        source_a, source_b, source_c, flow_field_a, flow_field_b, flow_field_c, mask_a, mask_b, mask_c = ctx.saved_tensors
        grad_source_a = Variable(source_a.new(source_a.size()).zero_())
        grad_source_b = Variable(source_b.new(source_b.size()).zero_())
        grad_source_c = Variable(source_c.new(source_c.size()).zero_())
        grad_flow_field_a = Variable(flow_field_a.new(flow_field_a.size()).zero_())
        grad_flow_field_b = Variable(flow_field_b.new(flow_field_b.size()).zero_())
        grad_flow_field_c = Variable(flow_field_c.new(flow_field_c.size()).zero_())

        mutil_block_extractor_cuda.backward(source_a, source_b, source_c, flow_field_a, flow_field_b, flow_field_c, mask_a, mask_b, mask_c, grad_output.data,
                                 grad_source_a.data, grad_source_b.data, grad_source_c.data, grad_flow_field_a.data, grad_flow_field_b.data, grad_flow_field_c.data,
                                 ctx.kernel_size)

        return grad_source_a, grad_source_b, grad_source_c, grad_flow_field_a, grad_flow_field_b, grad_flow_field_c, None, None, None, None


class MutilBlockExtractor(Module):
    def __init__(self, kernel_size=3):
        super(MutilBlockExtractor, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, source_a, source_b, source_c, flow_field_a, flow_field_b, flow_field_c, mask_a, mask_b, mask_c):
        source_a_c = source_a.contiguous()
        source_b_c = source_b.contiguous()
        source_c_c = source_c.contiguous()
        flow_field_a_c = flow_field_a.contiguous()
        flow_field_b_c = flow_field_b.contiguous()
        flow_field_c_c = flow_field_c.contiguous()
        mask_a_c = mask_a.contiguous()
        mask_b_c = mask_b.contiguous()
        mask_c_c = mask_c.contiguous()
        return MutilBlockExtractorFunction.apply(source_a_c, source_b_c, source_c_c, flow_field_a_c, flow_field_b_c, flow_field_c_c, 
                                        mask_a_c, mask_b_c, mask_c_c, self.kernel_size)
