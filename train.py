import argparse
import numpy as np
import os
import json
import h5py
import copy
import collections
import re
import datetime
import hashlib
import time
from timeit import default_timer

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--base_network', type=str, default='VTN',
                    help='Specifies the base network (either VTN or VoxelMorph)')
parser.add_argument('-n', '--n_cascades', type=int, default=1,
                    help='Number of cascades')
parser.add_argument('-r', '--rep', type=int, default=1,
                    help='Number of times of shared-weight cascading')
parser.add_argument('-g', '--gpu', type=str, default='0',
                    help='Specifies gpu device(s)')
parser.add_argument('-c', '--checkpoint', type=str, default=None,
                    help='Specifies a previous checkpoint to start with')
parser.add_argument('-d', '--dataset', type=str, default="datasets/liver.json",
                    help='Specifies a data config')
parser.add_argument('--batch', type=int, default=4,
                    help='Number of image pairs per batch')
parser.add_argument('--round', type=int, default=20000,
                    help='Number of batches per epoch')
parser.add_argument('--epochs', type=float, default=5,
                    help='Number of epochs')
parser.add_argument('--fast_reconstruction', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--val_steps', type=int, default=100)
parser.add_argument('--net_args', type=str, default='')
parser.add_argument('--data_args', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--clear_steps', action='store_true')
parser.add_argument('--finetune', type=str, default=None)
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--logs', type=str, default='')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import tensorflow as tf
import tflearn
import keras

import network
import data_util.liver
import data_util.brain
from data_util.data import Split

def main():
    repoRoot = os.path.dirname(os.path.realpath(__file__))

    if args.finetune is not None:
        args.clear_steps = True

    batchSize = args.batch
    iterationSize = args.round

    gpus = 0 if args.gpu == '-1' else len(args.gpu.split(','))

    Framework = network.FrameworkUnsupervised
    Framework.net_args['base_network'] = args.base_network
    Framework.net_args['n_cascades'] = args.n_cascades
    Framework.net_args['rep'] = args.rep
    #Framework.net_args['augmentation'] = 'identity'
    Framework.net_args.update(eval('dict({})'.format(args.net_args)))
    with open(os.path.join(args.dataset), 'r') as f:
        cfg = json.load(f)
        image_size = cfg.get('image_size', [128, 128, 128])
        image_type = cfg.get('image_type')
    framework = Framework(devices=gpus, image_size=image_size, segmentation_class_value=cfg.get('segmentation_class_value', None), fast_reconstruction = args.fast_reconstruction)
    Dataset = eval('data_util.{}.Dataset'.format(image_type))
    print('Graph built.')

    # load training set and validation set

    def set_tf_keys(feed_dict, **kwargs):
        ret = dict([(k + ':0', v) for k, v in feed_dict.items()])
        ret.update([(k + ':0', v) for k, v in kwargs.items()])
        return ret
    config = tf.ConfigProto(allow_soft_placement = True) 
    config.gpu_options.allow_growth = True
    

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=5, keep_checkpoint_every_n_hours=5)
        if args.checkpoint is None:
            steps = 0
            tf.global_variables_initializer().run()
        else:
            if '\\' not in args.checkpoint and '/' not in args.checkpoint:
                args.checkpoint = os.path.join(
                    repoRoot, 'weights', args.checkpoint)
            if os.path.isdir(args.checkpoint):
                args.checkpoint = tf.train.latest_checkpoint(args.checkpoint)

            tf.global_variables_initializer().run()
            checkpoints = args.checkpoint.split(';')
            if args.clear_steps:
                steps = 0
            else:
                steps = int(re.search('model-(\d+)', checkpoints[0]).group(1))
            def optimistic_restore(session, save_file):
                reader = tf.train.NewCheckpointReader(save_file)
                saved_shapes = reader.get_variable_to_shape_map()#get the saving model var
                var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                                if var.name.split(':')[0] in saved_shapes])
                restore_vars = []
                name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
                with tf.variable_scope('', reuse=True):
                    for var_name, saved_var_name in var_names:
                        curr_var = name2var[saved_var_name]
                        var_shape = curr_var.get_shape().as_list()
                        if var_shape == saved_shapes[saved_var_name]:
                            restore_vars.append(curr_var)
                saver_list = tf.train.Saver(restore_vars)
                saver_list.restore(session, save_file)
            for cp in checkpoints:
                optimistic_restore(sess, cp)

        data_args = eval('dict({})'.format(args.data_args))
        data_args.update(framework.data_args)
        print('data_args', data_args)
        dataset = Dataset(args.dataset, **data_args)
        if args.finetune is not None:
            if 'finetune-train-%s' % args.finetune in dataset.schemes:
                dataset.schemes[Split.TRAIN] = dataset.schemes['finetune-train-%s' %
                                                               args.finetune]
            if 'finetune-val-%s' % args.finetune in dataset.schemes:
                dataset.schemes[Split.VALID] = dataset.schemes['finetune-val-%s' %
                                                               args.finetune]
            print('train', dataset.schemes[Split.TRAIN])
            print('val', dataset.schemes[Split.VALID])
        generator = dataset.generator(Split.TRAIN, batch_size=batchSize, loop=True)

        if not args.debug:
            if args.finetune is not None:
                run_id = os.path.basename(os.path.dirname(args.checkpoint))
                if not run_id.endswith('_ft' + args.finetune):
                    run_id = run_id + '_ft' + args.finetune
            else:
                pad = ''
                retry = 1
                while True:
                    dt = datetime.datetime.now(
                        tz=datetime.timezone(datetime.timedelta(hours=8)))
                    run_id = dt.strftime('%b%d-%H%M') + pad
                    modelPrefix = os.path.join(repoRoot, 'weights', run_id)
                    try:
                        os.makedirs(modelPrefix)
                        break
                    except Exception as e:
                        print('Conflict with {}! Retry...'.format(run_id))
                        pad = '_{}'.format(retry)
                        retry += 1
            modelPrefix = os.path.join(repoRoot, 'weights', run_id)
            if not os.path.exists(modelPrefix):
                os.makedirs(modelPrefix)
            if args.name is not None:
                run_id += '_' + args.name
            if args.logs is None:
                log_dir = 'logs'
            else:
                log_dir = os.path.join('logs', args.logs)
            summary_path = os.path.join(repoRoot, log_dir, run_id)
            if not os.path.exists(summary_path):
                os.makedirs(summary_path)
            summaryWriter = tf.summary.FileWriter(summary_path, sess.graph)
            with open(os.path.join(modelPrefix, 'args.json'), 'w') as fo:
                json.dump(vars(args), fo)

        if args.finetune is not None:
            learningRates = [1e-5 / 2, 1e-5 / 2, 1e-5 / 2, 1e-5 / 4, 1e-5 / 8]
            #args.epochs = 1
        else:
            #learningRates = [1e-4, 1e-4, 1e-4, 1e-4 , 1e-4 / 2,1e-4 / 2, 1e-4 / 2, 1e-4 / 4, 1e-4 / 4,1e-4 / 8]
            learningRates = [1e-4, 1e-4, 8e-5,8e-5, 6e-5 , 6e-5, 4e-5, 4e-5, 2e-5, 2e-5,1e-5]

            # Training

        def get_lr(steps):
            m = args.lr / learningRates[0]
            return m * learningRates[steps // iterationSize]

        last_save_stamp = time.time()
        best_dice_score = 0.6
        while True:
            if hasattr(framework, 'get_lr'):
                lr = framework.get_lr(steps, batchSize)
            else:
                lr = get_lr(steps)
            t0 = default_timer()
            fd = next(generator)
            fd.pop('mask', [])
            fd.pop('id1', [])
            fd.pop('id2', [])
            t1 = default_timer()
            tflearn.is_training(True, session=sess)
            summ, _ = sess.run([framework.summaryExtra, framework.adamOpt],
                               set_tf_keys(fd, learningRate=lr))
            
            for v in tf.Summary().FromString(summ).value:
                if v.tag == 'loss':
                    loss = v.simple_value

            steps += 1
            if args.debug or steps % 10 == 0:
                if steps >= args.epochs * iterationSize:
                    break

                if not args.debug:
                    summaryWriter.add_summary(summ, steps)

                if steps % 50 == 0:
                    if hasattr(framework, 'summaryImages'):
                        summ, = sess.run([framework.summaryImages],
                                         set_tf_keys(fd))
                        summaryWriter.add_summary(summ, steps)

                if steps % 50 == 0:
                    print('*%s* ' % run_id,
                          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                          'Steps %d, Total time %.2f, data %.2f%%. Loss %.3e lr %.3e' % (steps,
                                                                                         default_timer() - t0,
                                                                                         (t1 - t0) / (
                                                                                             default_timer() - t0),
                                                                                         loss,
                                                                                         lr),
                          end='\n')


                if args.debug or steps % args.val_steps == 0:
                    try:
                        val_gen = dataset.generator(
                            Split.VALID, loop=False, batch_size=1)
                        metrics = framework.validate(
                            sess, val_gen, summary=True)
                        val_summ = tf.Summary(value=[
                            tf.Summary.Value(tag='val_' + k, simple_value=v) for k, v in metrics.items()
                        ])
                        print('dice:',metrics['dice_score'])
                        print('ncc:',metrics['total_ncc'])
                        if metrics['dice_score']>best_dice_score:
                            best_dice_score = metrics['dice_score']
                            print('saving best dice sore:{}'.format(best_dice_score))
                            saver.save(sess, os.path.join(modelPrefix, 'model'),global_step=steps,write_meta_graph=False)
                            with open(os.path.join(modelPrefix,'log.txt'),'a+') as f:
                                f.write('saving best dice sore:{},steps={} \n'.format(best_dice_score,steps))
                        summaryWriter.add_summary(val_summ, steps)
                    except:
                        if steps == args.val_steps:
                            print('Step {}, validation failed!'.format(steps))
    print('Finished.')


if __name__ == '__main__':
    main()
