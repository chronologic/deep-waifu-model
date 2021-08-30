import io
import json                    
import base64                  
import logging             
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, abort, make_response
import argparse
from UGATIT import UGATIT
from utils import *

app = Flask(__name__)          
app.logger.setLevel(logging.DEBUG)

def parse_args():
    desc = "Tensorflow implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='runner', help='[train / test / web / runner]')
    parser.add_argument('--light', type=str2bool, default=False,
                        help='[U-GAT-IT full version / U-GAT-IT light version]')
    parser.add_argument('--dataset', type=str, default='selfie2anime', help='dataset_name')

    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=10000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
    parser.add_argument('--decay_epoch', type=int, default=50, help='decay epoch')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--GP_ld', type=int, default=10, help='The gradient penalty lambda')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight about GAN')
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight about Cycle')
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight about Identity')
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight about CAM')
    parser.add_argument('--gan_type', type=str, default='lsgan',
                        help='[gan / lsgan / wgan-gp / wgan-lp / dragan / hinge]')

    parser.add_argument('--smoothing', type=str2bool, default=True, help='AdaLIN smoothing effect')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')
    parser.add_argument('--n_critic', type=int, default=1, help='The number of critic')
    parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--augment_flag', type=str2bool, default=True, help='Image augmentation use or not')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args



sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU': 0}))
args = parse_args()
gan = UGATIT(sess, args)

# build graph
gan.build_model()

# show network architecture
show_all_variables()

gan.test_endpoint_init()


@app.route("/selfie2anime", methods=['POST'])
def selfie2anime():         
    file = request.files['file']

    # convert string of image data to uint8
    nparr = np.fromfile(file, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # open session
    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True), device_count = {'GPU': 0}) as sess:
    #     gan = UGATIT(sess, args)

    #     # build graph
    #     gan.build_model()

    #     # show network architecture
    #     show_all_variables()

        # do some fancy processing here....
    fake_img = gan.test_endpoint(img)

    # save the file with to our photos folder
    # filename = str(uuid.uuid1()) + '.png'
    # cv2.imwrite('uploads/' + filename, fake_img)
    # # append image urls
    # file_urls.append(photos.url(filename))
    retval, buffer = cv2.imencode('.png', fake_img)
    response = make_response(buffer.tobytes())
    response.headers['Content-Type'] = 'image/png'

    return response

  
  
def run_server_api():
    app.run(host='0.0.0.0', port=8080)
  
  
if __name__ == "__main__":     
    run_server_api()