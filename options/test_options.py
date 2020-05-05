from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self,  parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here')
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(phase='test')
        parser.set_defaults(batchSize=1)
        parser.add_argument('--calcfid', action='store_true', help='calc fid')
        self.isTrain = False

        return parser
