import argparse
import os
import json
import re
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from PIL import Image
import math
import scipy.misc
import xlwt

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, default=None,
                    help='Specifies a previous checkpoint to load')
parser.add_argument('-r', '--rep', type=int, default=1,
                    help='Number of times of shared-weight cascading')
parser.add_argument('-g', '--gpu', type=str, default='0',
                    help='Specifies gpu device(s)')
parser.add_argument('-d', '--dataset', type=str, default=None,
                    help='Specifies a data config')
parser.add_argument('-v', '--val_subset', type=str, default=None)
parser.add_argument('--batch', type=int, default=4, help='Size of minibatch')
parser.add_argument('--fast_reconstruction', action='store_true')
parser.add_argument('--paired', action='store_true')
parser.add_argument('--data_args', type=str, default=None)
parser.add_argument('--net_args', type=str, default=None)
parser.add_argument('--name', type=str, default=None)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import tensorflow as tf
import tflearn

import network
import data_util.liver
import data_util.brain


def main():
    if args.checkpoint is None:
        print('Checkpoint must be specified!')
        return
    if ':' in args.checkpoint:
        args.checkpoint, steps = args.checkpoint.split(':')
        steps = int(steps)
        print(steps)
    else:
        steps = None
    args.checkpoint = find_checkpoint_step(args.checkpoint, steps)
    print(args.checkpoint)
    model_dir = os.path.dirname(args.checkpoint)
    try:
        with open(os.path.join(model_dir, 'args.json'), 'r') as f:
            model_args = json.load(f)
        print(model_args)
    except Exception as e:
        print(e)
        model_args = {}

    if args.dataset is None:
        args.dataset = model_args['dataset']
    if args.data_args is None:
        args.data_args = model_args['data_args']

    Framework = network.FrameworkUnsupervised
    Framework.net_args['base_network'] = model_args['base_network']
    Framework.net_args['n_cascades'] = model_args['n_cascades']
    Framework.net_args['rep'] = args.rep
    Framework.net_args['augmentation'] = 'identity'
    Framework.net_args.update(eval('dict({})'.format(model_args['net_args'])))
    if args.net_args is not None:
        Framework.net_args.update(eval('dict({})'.format(args.net_args)))
    with open(os.path.join(args.dataset), 'r') as f:
        cfg = json.load(f)
        image_size = cfg.get('image_size', [160, 160, 160])
        image_type = cfg.get('image_type')
    gpus = 0 if args.gpu == '-1' else len(args.gpu.split(','))
    framework = Framework(devices=gpus, image_size=image_size, segmentation_class_value=cfg.get(
        'segmentation_class_value', None), fast_reconstruction=args.fast_reconstruction, validation=True)
    print('Graph built')

    Dataset = eval('data_util.{}.Dataset'.format(image_type))
    ds = Dataset(args.dataset, batch_size=args.batch, paired=args.paired, **
                 eval('dict({})'.format(args.data_args)))
                 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    tf.global_variables_initializer().run(session=sess)

    checkpoint = args.checkpoint
    checkpoint = args.checkpoint
    saver = tf.train.Saver(tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES))
    saver.restore(sess, checkpoint)

    tflearn.is_training(False, session=sess)

    val_subsets = [data_util.liver.Split.VALID]
    if args.val_subset is not None:
        val_subsets = args.val_subset.split(',')
    
    tflearn.is_training(False, session=sess)
    writebook = xlwt.Workbook()  
    testSheet1= writebook.add_sheet('dice')
    keys = ['total_ncc','jaccs','landmark_dists','dices','det_A']
    #'image_fixed','warped_moving','seg_fixed','warped_seg_moving','jaccs','landmark_dists','jacobian_det',
    if not os.path.exists('evaluate'):
        os.mkdir('evaluate')
    path_prefix = os.path.join('evaluate', short_name(checkpoint))
    if args.rep > 1:
        path_prefix = path_prefix + '-rep' + str(args.rep)
    if args.name is not None:
        path_prefix = path_prefix + '-' + args.name
    for val_subset in val_subsets:
        if args.val_subset is not None:
            output_fname = path_prefix + '-' + str(val_subset) + '.txt'
        else:
            output_fname = path_prefix + '.txt'
            output_xls = path_prefix + '.xls'
        with open(output_fname, 'w') as fo:
            print("Validation subset {}".format(val_subset))
            gen = ds.generator(val_subset, loop=False)
            results = framework.validate(sess, gen, keys=keys, summary=False, predict=True, show_tqdm=True)
            ##################image save#########################
            image_save_path = os.path.join('./test_images', short_name(checkpoint))#+'_onebyone'
            if not os.path.isdir(image_save_path):
                os.makedirs(image_save_path)

            ''' for i in range(len(results['image_fixed'])):
                print(results['id2'][i])
                writer = sitk.ImageFileWriter()

                sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(results['image_fixed'][i][:,:,:,0])), image_save_path+'/'+results['id2'][i]+'_fixed.mhd', True)
                sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(results['warped_moving'][i][:,:,:,0])), image_save_path+'/'+results['id2'][i]+'_moving.mhd', True)
                sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(results['moving_img'][i][:,:,:,0])), image_save_path+'/'+results['id2'][i]+'_moving_raw.mhd', True)
                sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(results['real_flow'][i][:,:,:,:])), image_save_path+'/'+results['id2'][i]+'_flow.mhd', True)
                
                warped_seg = np.squeeze(np.zeros(results['warped_seg_moving'][i][:,:,:,0].shape))
                seg_fixed = np.squeeze(np.zeros(results['warped_seg_moving'][i][:,:,:,0].shape))

                for seg in range(results['warped_seg_moving'][i].shape[-1]):
                    sub_warp = np.squeeze((results['warped_seg_moving'][i])[:,:,:,seg])
                    sub_warp = np.where(sub_warp>127.5,seg+1,0)
                    sub_seg = np.squeeze((results['seg_fixed'][i])[:,:,:,seg])
                    sub_seg = np.where(sub_seg>127.5,seg+1,0)
                    warped_seg += sub_warp
                    seg_fixed += sub_seg
                sitk.WriteImage(sitk.GetImageFromArray(warped_seg), image_save_path+'/'+results['id2'][i]+'_warped_seg.mhd', True)
                sitk.WriteImage(sitk.GetImageFromArray(seg_fixed), image_save_path+'/'+results['id2'][i]+'_seg_fixed.mhd', True)  '''

            
            for i in range(len(results['dices'])):
                print(results['id1'][i],results['id2'][i],np.mean(results['dices'][i]),np.mean(results['jaccs'][i]), np.mean(results['landmark_dists'][i]),results['det_A'][i], file=fo)#
            writebook.save(output_xls) 
            print('Summary', file=fo)
            jaccs, dices, landmarks,ncc = results['jaccs'], results['dices'], results['landmark_dists'],results['total_ncc']
            print("Dice score: {} ({})".format(np.mean(dices), np.std(
                np.mean(dices, axis=-1))), file=fo)
            print("Jacc score: {} ({})".format(np.mean(jaccs), np.std(
                np.mean(jaccs, axis=-1))), file=fo)
            print("ncc score: {} ({})".format(np.mean(ncc), np.std(
                np.mean(ncc, axis=-1))), file=fo)
            print("Landmark distance: {} ({})".format(np.mean(landmarks), np.std(
                np.mean(landmarks, axis=-1))), file=fo)
            ########################################
            for seg in range(results['dices'].shape[1]):
                print("dice score for seg {}: {}".format(seg, np.mean(
                results['dices'][:,seg])), file=fo)
            ###########################################

def cbimage(img1, img2):
    shape = img1.shape
    num = 20
    grid = np.zeros(shape)
    bg_p_x = [int(shape[0]/num*i) for i in range(num)]
    bg_p_y = [int(shape[1]/num*i) for i in range(num)]
    for i in range(0, num, 2):
        for j in range(0, num, 2):
            grid[bg_p_x[i]:bg_p_x[i]+shape[0]//num,bg_p_y[j]:bg_p_y[j]+shape[1]//num] = 1
            grid[bg_p_x[i+1]:bg_p_x[i+1]+shape[0]//num,bg_p_y[j+1]:bg_p_y[j+1]+shape[1]//num] = 1
    img1_grid = img1*grid
    img2_grid = img2*(1-grid)
    cbimage = img1_grid+img2_grid
    return cbimage

def short_name(checkpoint):
    cpath, steps = os.path.split(checkpoint)
    _, exp = os.path.split(cpath)
    return exp + '-' + steps

def RenderFlow(flow, coef = 15, channel = (0, 1, 2), thresh = 1):
    flow = flow[:, :, 64]
    im_flow = np.stack([flow[:, :, c] for c in channel], axis = -1)
    #im_flow = 0.5 + im_flow / coef
    '''im_flow = np.abs(im_flow)
    im_flow = np.exp(-im_flow / coef)
    im_flow = im_flow * thresh
    #im_flow = 1 - im_flow / 20
    '''
    return im_flow
def find_checkpoint_step(checkpoint_path, target_steps=None):
    pattern = re.compile(r'model-(\d+).index')
    checkpoints = []
    for f in os.listdir(checkpoint_path):
        m = pattern.match(f)
        if m:
            steps = int(m.group(1))
            checkpoints.append((-steps if target_steps is None else abs(
                target_steps - steps), os.path.join(checkpoint_path, f.replace('.index', ''))))
    return min(checkpoints, key=lambda x: x[0])[1]


if __name__ == '__main__':
    main()
