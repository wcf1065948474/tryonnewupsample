import time
from options.train_options import TrainOptions
import data as Dataset
from model import create_model
import util.util as util
# from util.visualizer import Visualizer
full_time = 9*3600
max_epoch_time = 0

if __name__ == '__main__':
    # get training options
    opt = TrainOptions().parse()
    # create a dataset
    dataset = Dataset.create_dataloader(opt)
    dataset_size = len(dataset) * opt.batchSize
    print('training images = %d' % dataset_size)
    # create a model
    model = create_model(opt)
    # model = model.to()  
    # create a visualizer
    # visualizer = Visualizer(opt)
    # training flag
    keep_training = True
    max_iteration = opt.niter+opt.niter_decay
    epoch = opt.which_iter
    total_iteration = opt.iter_count

    # training process
    while(keep_training):
        epoch_start_time = time.time()
        epoch+=1
        print('\n Training epoch: %d' % epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_iteration += 1
            model.set_input(data)
            model.optimize_parameters()

            # display images on visdom and save images
            if total_iteration % opt.display_freq == 0:
                vis = model.get_current_visuals()
                for img_key in vis.keys():
                    util.save_image(vis[img_key],'{}/{}.jpg'.format('result/vis',img_key+str(total_iteration)))
                if hasattr(model, 'distribution'):
                    visualizer.plot_current_distribution(model.get_current_dis()) 

            # print training loss and save logging information to the disk
            if total_iteration % opt.print_freq == 0:
                losses = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                loss = ''
                for k in losses.keys():
                    loss = loss + k + str(losses[k])
                print('epoch={},total={},loss={},time={}'.format(epoch, total_iteration, loss, t))


        #     if total_iteration % opt.eval_iters_freq == 0:
        #         model.eval() 
        #         if hasattr(model, 'eval_metric_name'):
        #             eval_results = model.get_current_eval_results()  
        #             visualizer.print_current_eval(epoch, total_iteration, eval_results)
        #             if opt.display_id > 0:
        #                 visualizer.plot_current_score(total_iteration, eval_results)
                    

        epoch_time = time.time() - epoch_start_time
        full_time -= epoch_time
        if epoch_time > max_epoch_time:
            max_epoch_time = epoch_time
        if full_time < max_epoch_time:
            keep_training = False
        model.save_networks(epoch)
        model.update_learning_rate()
    print('\nEnd training')
