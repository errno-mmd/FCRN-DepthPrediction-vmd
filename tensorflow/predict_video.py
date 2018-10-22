import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import logging
import cv2
import datetime
import os
import re
import shutil
import imageio
import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

level = {0:logging.ERROR,
            1:logging.WARNING,
            2:logging.INFO,
            3:logging.DEBUG}

def predict_video(model_path, video_path, baseline_path, interval, smoothed_2d, start_frame):
    logger.info("深度推定出力開始")

    # 深度用サブディレクトリ
    subdir = '{0}/depth'.format(baseline_path)
    if os.path.exists(subdir):
        # 既にディレクトリがある場合、一旦削除
        shutil.rmtree(subdir)
    os.makedirs(subdir)

    #関節位置情報ファイル
    depthf = open(baseline_path +'/depth.txt', 'w')

    # Default input size
    height = 288
    width = 512
    channels = 3
    batch_size = 1
    scale = 0

    # # tensorflowをリセットする
    # tf.reset_default_graph()

    # 映像サイズを取得する
    n = 0
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        orig_width = cap.get(3)   # float
        orig_height = cap.get(4) # float
        logger.debug("width: {0}, height: {1}".format(orig_width, orig_height))
        
        # 縮小倍率
        scale = width / orig_width

        logger.debug("scale: {0}".format(scale))

        height = int(orig_height * scale)

        logger.debug("width: {0}, height: {1}".format(width, height))

        break

    # 再設定したサイズでtensorflow準備
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

    png_lib = []

    with tf.Session() as sess:

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_path)

        # 動画を1枚ずつ画像に変換する
        n = 0
        cap = cv2.VideoCapture(video_path)
        while(cap.isOpened()):
            # 動画から1枚キャプチャして読み込む
            flag, frame = cap.read()  # Capture frame-by-frame
            # キャプチャが終わっていたら終了
            if flag == False:  # Is a frame left?
                break

            # 開始フレームより前は飛ばす
            if start_frame - interval >= n:
                n += 1
                continue

            if n % interval == 0:
                # 先に間引き分同じのを追加
                if interval > 1 and n > start_frame:
                    for m in range(interval - 1):
                        # logger.debug("間引き分追加 {0}".format(m))
                        png_lib.append(imageio.imread("{0}/depth_{1:012d}.png".format(subdir, n - interval)))

                # 一定間隔フレームおきにキャプチャした画像を深度推定する
                logger.info("深度推定: n={0}".format(n))

                # smoothedのindex
                s_idx = n - start_frame

                # キャプチャ画像を読み込む
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = img.resize([width,height], Image.ANTIALIAS)
                img = np.array(img).astype('float32')
                img = np.expand_dims(np.asarray(img), axis = 0)
                    
                # Use to load from npy file
                #net.load(model_path, sess) 

                # Evalute the network for the given image
                pred = sess.run(net.get_output(), feed_dict={input_node: img})

                # 深度解析後の画像サイズ
                pred_height = len(pred[0])
                pred_width = len(pred[0][0])

                logger.debug("smoothed_2d[s_idx] {0}".format(smoothed_2d[s_idx]))

                # 腰 ------------------------

                # 両足の付け根の中間を取得する
                waist_smoothed_center_x = np.average([smoothed_2d[s_idx][0][0], smoothed_2d[s_idx][1][0]])
                waist_smoothed_center_y = np.average([smoothed_2d[s_idx][0][1], smoothed_2d[s_idx][1][1]])

                logger.debug("waist_smoothed_center_x: {0}, waist_smoothed_center_y: {1}".format(waist_smoothed_center_x, waist_smoothed_center_y))

                # オリジナルの画像サイズから、縮尺を取得
                waist_scale_orig_x = waist_smoothed_center_x / orig_width
                waist_scale_orig_y = waist_smoothed_center_y / orig_height

                logger.debug("waist_scale_orig_x: {0}, waist_scale_orig_y: {1}".format(waist_scale_orig_x, waist_scale_orig_y))

                # 縮尺を展開して、深度解析後の画像サイズに合わせる
                waist_pred_x = int(pred_width * waist_scale_orig_x)
                waist_pred_y = int(pred_height * waist_scale_orig_y)

                logger.debug("waist_pred_x: {0}, waist_pred_y: {1}, depth: {2}".format(waist_pred_x, waist_pred_y, pred[0][waist_pred_y][waist_pred_x][0]))

                # 右足首 ------------------------

                # 右足首を取得する
                right_ankle_smoothed_center_x = smoothed_2d[s_idx][2][0]
                right_ankle_smoothed_center_y = smoothed_2d[s_idx][2][1]

                logger.debug("right_ankle_smoothed_center_x: {0}, right_ankle_smoothed_center_y: {1}".format(right_ankle_smoothed_center_x, right_ankle_smoothed_center_y))

                # オリジナルの画像サイズから、縮尺を取得
                right_ankle_scale_orig_x = right_ankle_smoothed_center_x / orig_width
                right_ankle_scale_orig_y = right_ankle_smoothed_center_y / orig_height

                logger.debug("right_ankle_scale_orig_x: {0}, right_ankle_scale_orig_y: {1}".format(right_ankle_scale_orig_x, right_ankle_scale_orig_y))

                # 縮尺を展開して、深度解析後の画像サイズに合わせる
                right_ankle_pred_x = int(pred_width * right_ankle_scale_orig_x)
                right_ankle_pred_y = int(pred_height * right_ankle_scale_orig_y)

                logger.debug("right_ankle_pred_x: {0}, right_ankle_pred_y: {1}, depth: {2}".format(right_ankle_pred_x, right_ankle_pred_y, pred[0][right_ankle_pred_y][right_ankle_pred_x][0]))

                # 左足首 ------------------------

                # 左足首を取得する
                left_ankle_smoothed_center_x = smoothed_2d[s_idx][3][0]
                left_ankle_smoothed_center_y = smoothed_2d[s_idx][3][1]

                logger.debug("left_ankle_smoothed_center_x: {0}, left_ankle_smoothed_center_y: {1}".format(left_ankle_smoothed_center_x, left_ankle_smoothed_center_y))

                # オリジナルの画像サイズから、縮尺を取得
                left_ankle_scale_orig_x = left_ankle_smoothed_center_x / orig_width
                left_ankle_scale_orig_y = left_ankle_smoothed_center_y / orig_height

                logger.debug("left_ankle_scale_orig_x: {0}, left_ankle_scale_orig_y: {1}".format(left_ankle_scale_orig_x, left_ankle_scale_orig_y))

                # 縮尺を展開して、深度解析後の画像サイズに合わせる
                left_ankle_pred_x = int(pred_width * left_ankle_scale_orig_x)
                left_ankle_pred_y = int(pred_height * left_ankle_scale_orig_y)

                logger.debug("left_ankle_pred_x: {0}, left_ankle_pred_y: {1}, depth: {2}".format(left_ankle_pred_x, left_ankle_pred_y, pred[0][left_ankle_pred_y][left_ankle_pred_x][0]))

                # 出力 ------------------------

                # 深度ファイルに出力
                depthf.write("{0}, {1}, {2}, {3}\n".format(n, pred[0][waist_pred_y][waist_pred_x][0], pred[0][right_ankle_pred_y][right_ankle_pred_x][0], pred[0][left_ankle_pred_y][left_ankle_pred_x][0]))

                # Plot result
                plt.cla()
                plt.clf()
                ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
                plt.colorbar(ii)

                # 散布図のようにして、出力に使ったポイントを明示
                plt.scatter(waist_pred_x, waist_pred_y, s=5, c="#FFFFFF")
                plt.scatter(right_ankle_pred_x, right_ankle_pred_y, s=5, c="#FFFFFF")
                plt.scatter(left_ankle_pred_x, left_ankle_pred_y, s=5, c="#FFFFFF")

                # 深度画像保存
                plotName = "{0}/depth_{1:012d}.png".format(subdir, n)
                plt.savefig(plotName)
                logger.debug("Save: {0}".format(plotName))

                # アニメーションGIF用に保持
                png_lib.append(imageio.imread(plotName))

                plt.close()

            n += 1

    logger.info("creating Gif {0}/movie_depth.gif, please Wait!".format(baseline_path))
    imageio.mimsave('{0}/movie_depth.gif'.format(baseline_path), png_lib, fps=30)

    # 終わったら後処理
    cap.release()
    cv2.destroyAllWindows()

    logger.info("Done!!")
    logger.info("深度推定結果: {0}".format(subdir))

def predict(model_data_path, image_path):

    
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
   
    # Read image
    img = Image.open(image_path)
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)
   
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        #net.load(model_data_path, sess) 

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        
        # Plot result
        fig = plt.figure()
        ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        fig.colorbar(ii)
        plt.show()
        
        return pred
        

# 関節2次元情報を取得
def load_smoothed_2d(smoothed_file):
    smoothed_2d = []
    n = 0
    with open(smoothed_file, "r") as sf:
        line = sf.readline() # 1行を文字列として読み込む(改行文字も含まれる)
        
        while line:
            # 空白で複数項目に分解
            smoothed = re.split("\s+", line)

            # logger.debug(smoothed)

            smoothed_2d.append([ \
                # 右足付け根
                [float(smoothed[16]), float(smoothed[17])], \
                # 左足付け根
                [float(smoothed[22]), float(smoothed[23])], \
                # 右足首
                [float(smoothed[20]), float(smoothed[21])], \
                # 左足首
                [float(smoothed[26]), float(smoothed[27])] \
            ])

            n += 1

            line = sf.readline()
    
    return smoothed_2d
       
# 開始フレームを取得
def load_start_frame(start_frame_file):
    n = 0
    with open(start_frame_file, "r") as sf:
        return int(sf.readline())

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', dest='model_path', help='Converted parameters for the model', type=str)
    parser.add_argument('--video_path', dest='video_path', help='input video', type=str)
    parser.add_argument('--baseline_path', dest='baseline_path', help='baseline result path', type=str)
    parser.add_argument('--interval', dest='interval', help='interval', type=int)
    parser.add_argument('--verbose', dest='verbose', help='verbose', type=int)
    args = parser.parse_args()

    logger.setLevel(level[args.verbose])

    # 関節二次元データを取得
    smoothed_2d = load_smoothed_2d("{0}/smoothed.txt".format(args.baseline_path))

    # 開始フレームインデックス
    start_frame = load_start_frame("{0}/start_frame.txt".format(args.baseline_path))
    logger.info("開始フレームインデックス: %d", start_frame)

    # 間隔は1以上の整数
    interval = args.interval if args.interval > 0 else 1

    # Predict the image
    predict_video(args.model_path, args.video_path, args.baseline_path, interval, smoothed_2d, start_frame)

if __name__ == '__main__':
    main()

        



