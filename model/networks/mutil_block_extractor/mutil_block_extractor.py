from torch.nn.modules.module import Module
from torch.autograd import Function, Variable
import mutil_block_extractor_cuda

class MutilBlockExtractorFunction(Function):

    @staticmethod
    def forward(ctx, source_a, source_b, source_c, flow_field_a, flow_field_b, flow_field_c, masks_a, masks_b, masks_c, kernel_size):
        assert source_a.is_contiguous()
        assert flow_field_a.is_contiguous()
        assert masks_a.is_contiguous()

        # TODO: check the shape of the inputs 
        bs, ds, hs, ws = source_a.size()
        bf, df, hf, wf = flow_field_a.size()
        # assert bs==bf and hs==hf and ws==wf
        assert df==2

        ctx.save_for_backward(source_a, source_b, source_c, flow_field_a, flow_field_b, flow_field_c, masks_a, masks_b, masks_c)
        ctx.kernel_size = kernel_size

        output = flow_field_a.new(bs, ds, kernel_size*hf, kernel_size*wf).zero_()

        if not source_a.is_cuda:
            raise NotImplementedError
        else:
            mutil_block_extractor_cuda.forward(source_a, source_b, source_c, flow_field_a, flow_field_b, flow_field_c, masks_a, masks_b, masks_c, output, kernel_size)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output.contiguous()
        source_a, source_b, source_c, flow_field_a, flow_field_b, flow_field_c, masks_a, masks_b, masks_c = ctx.saved_tensors
        grad_source_a = Variable(source_a.new(source_a.size()).zero_())
        grad_source_b = Variable(source_b.new(source_b.size()).zero_())
        grad_source_c = Variable(source_c.new(source_c.size()).zero_())
        grad_flow_field_a = Variable(flow_field_a.new(flow_field_a.size()).zero_())
        grad_flow_field_b = Variable(flow_field_b.new(flow_field_b.size()).zero_())
        grad_flow_field_c = Variable(flow_field_c.new(flow_field_c.size()).zero_())

        mutil_block_extractor_cuda.backward(source_a, source_b, source_c, flow_field_a, flow_field_b, flow_field_c, masks_a, masks_b, masks_c, grad_output.data,
                                 grad_source_a.data, grad_source_b.data, grad_source_c.data, grad_flow_field_a.data, grad_flow_field_b.data, grad_flow_field_c.data,
                                 ctx.kernel_size)

        return grad_source_a, grad_source_b, grad_source_c, grad_flow_field_a, grad_flow_field_b, grad_flow_field_c, None, None, None, None


class MutilBlockExtractor(Module):
    def __init__(self, kernel_size=3, batchsize=4):
        super(MutilBlockExtractor, self).__init__()
        self.kernel_size = kernel_size
        self.batchsize = batchsize

    def forward(self, source, flow_field, masks):
        source_a = source[0].contiguous()
        source_b = source[1].contiguous()
        source_c = source[2].contiguous()

        flow_field_a = flow_field[0].contiguous()
        flow_field_b = flow_field[1].contiguous()
        flow_field_c = flow_field[2].contiguous()

        masks_a = masks[0].contiguous()
        masks_b = masks[1].contiguous()
        masks_c = masks[2].contiguous()
        return MutilBlockExtractorFunction.apply(source_a, source_b, source_c, flow_field_a, flow_field_b, flow_field_c, masks_a, masks_b , masks_c, self.kernel_size)