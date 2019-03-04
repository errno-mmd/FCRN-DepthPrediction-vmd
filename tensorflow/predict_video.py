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
import json
import copy
import sys
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ファイル出力ログ用
file_logger = logging.getLogger("message")

level = {0: logging.ERROR,
            1: logging.WARNING,
            2: logging.INFO,
            3: logging.DEBUG}


def predict_video(now_str, model_path, video_path, depth_path, interval, openpose_output_dir, openpose_2d, openpose_filenames, start_frame, reverse_frame_dict, order_specific_dict, is_avi_output, verbose):
    # 深度用サブディレクトリ
    subdir = '{0}/depth'.format(depth_path)
    if os.path.exists(subdir):
        # 既にディレクトリがある場合、一旦削除
        shutil.rmtree(subdir)
    os.makedirs(subdir)

    # ファイル用ログの出力設定
    log_file_path = '{0}/message.log'.format(depth_path)
    logger.debug(log_file_path)
    file_logger.addHandler(logging.FileHandler(log_file_path))
    file_logger.warn("深度推定出力開始 now: %s ---------------------------", now_str)

    logger.addHandler(logging.FileHandler('{0}/{1}.log'.format(depth_path, __name__)))

    # Default input size
    height = 288
    width = 512
    channels = 3
    batch_size = 1
    scale = 0

    # tensorflowをリセットする
    tf.reset_default_graph()

    # 映像サイズを取得する
    n = 0
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        orig_width = cap.get(3)   # float
        orig_height = cap.get(4)  # float
        logger.debug("width: {0}, height: {1}".format(orig_width, orig_height))

        # 縮小倍率
        scale = width / orig_width

        logger.debug("scale: {0}".format(scale))

        height = int(orig_height * scale)

        logger.debug("width: {0}, height: {1}".format(width, height))

        break

    # 再設定したサイズでtensorflow準備
    # Create a placeholder for the input image
    input_node = tf.placeholder(
        tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

    png_lib = []

    with tf.Session() as sess:

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        # 人数分の深度データ
        pred_multi_ary = [
            [[] for y in range(len(openpose_2d[0]))] for x in range(len(openpose_2d))]
        # 人数
        people_size = len(openpose_2d[0])
        # 並べ直したindex用配列
        sorted_idxs = [[-1 for y in range(people_size)]
                                          for x in range(len(openpose_filenames))]
        # 並べ直したindex用配列反転有無
        is_all_reverses = [[False for y in range(
            people_size)] for x in range(len(openpose_filenames))]
        # 並べ直したindex用配列反転有無(上半身のみ)
        is_upper_reverses = [[False for y in range(
            people_size)] for x in range(len(openpose_filenames))]
        # 並べ直したindex用配列反転有無(下半身のみ)
        is_lower_reverses = [[False for y in range(
            people_size)] for x in range(len(openpose_filenames))]
        # 並べ直したindex用片寄せ有無
        is_onesides = [[False for y in range(people_size)] for x in range(
            len(openpose_filenames))]
        # 現在反転中か否か(上半身)
        is_now_upper_reversed = [False for y in range(people_size)]
        # 現在反転中か否か(下半身)
        is_now_lower_reversed = [False for y in range(people_size)]
        # ファイル名フォーマット
        first_file_name = openpose_filenames[0]

        # 基準となる深度(1人目の0F目平均値)
        base_depth = 0.0

        # 動画を1枚ずつ画像に変換する
        cnt = 0
        cap = cv2.VideoCapture(video_path)
        while(cap.isOpened()):
            # 動画から1枚キャプチャして読み込む
            flag, frame = cap.read()  # Capture frame-by-frame

            # logger.debug("start_frame: %s, n: %s, len(openpose_2d): %s", start_frame, n, len(openpose_2d))

            # 深度推定のindex
            _idx = cnt - start_frame
            _display_idx = cnt - interval

            # 開始フレームより前は飛ばす
            if start_frame > cnt:
                cnt += 1
                continue

            # 終わったフレームより後は飛ばす
            if _idx >= len(openpose_2d) + interval:
                break

            if flag == True and ((_idx % interval == 0 and _idx < len(openpose_2d)) or (_idx == len(openpose_2d) - 1)):
                # 先に間引き分同じのを追加(間を埋める)
                if _idx % interval == 0 and level[verbose] <= logging.INFO and interval > 1 and cnt > start_frame:
                        for m in range(interval - 1):
                            # logger.debug("間引き分追加 {0}".format(m))
                            png_lib.append(imageio.imread(
                                "{0}/depth_{1:012d}.png".format(subdir, _display_idx)))

                # 一定間隔フレームおきにキャプチャした画像を深度推定する
                if _idx % (interval * 10) == 0:
                    logger.info("深度推定 idx: %s(%s)", _idx, cnt)

                # キャプチャ画像を読み込む
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = img.resize([width, height], Image.ANTIALIAS)
                img = np.array(img).astype('float32')
                img = np.expand_dims(np.asarray(img), axis=0)

                # Use to load from npy file
                # net.load(model_path, sess)

                # Evalute the network for the given image
                pred = sess.run(net.get_output(), feed_dict={input_node: img})

                # 深度解析後の画像サイズ
                pred_height = len(pred[0])
                pred_width = len(pred[0][0])

                # logger.debug("openpose_2d[_idx] {0}".format(openpose_2d[_idx]))

                # 深度散布図用リスト
                pred_multi_xys = [[] for x in range(people_size)]

                # 1フレーム分のデータ
                for xy_idx, one_xy_2d in enumerate(openpose_2d[_idx]):
                    # 深度リスト
                    pred_ary = []
                    pred_xy = []

                    # 一人分のデータ
                    for n_idx in range(len(one_xy_2d)):
                        # 関節位置を取得する
                        openpose_x = openpose_2d[_idx][xy_idx][n_idx][0]
                        openpose_y = openpose_2d[_idx][xy_idx][n_idx][1]

                        # logger.debug("s: %s, xy: %s, n: %s, openpose_x: %s, openpose_y: %s", _idx, xy_idx, n_idx, openpose_x, openpose_y)

                        # if openpose_x < 0 and openpose_y < 0:
                        #     if _idx > 0 and _idx % interval == 0:
                        #         # 信頼度が足りず、あり得ない値が設定されている場合、
                        #         # intevalの範囲内で関節位置が信頼できるものになるまで遡る
                        #         p_idx = 0
                        #         while openpose_x <= 0 and openpose_y <= 0 and _idx > p_idx:
                        #             p_idx += 1

                        #             if p_idx >= interval:
                        #                 break

                        #             # 関節位置を取得する
                        #             openpose_x = openpose_2d[_idx - p_idx][sorted_idxs[_idx - p_idx][xy_idx]][n_idx][0]
                        #             openpose_y = openpose_2d[_idx - p_idx][sorted_idxs[_idx - p_idx][xy_idx]][n_idx][1]

                        #         # logger.debug("pred_multi_ary: %s", pred_multi_ary)
                        #         # logger.debug("_idx:%s, xy_idx: %s, n_idx: %s, sorted: %s, len(pred_multi_ary):%s", _idx, xy_idx, n_idx, sorted_idxs[_idx - p_idx][xy_idx], len(pred_multi_ary[_idx - p_idx][sorted_idxs[_idx - p_idx][xy_idx]]))

                        #         # pred_ary.append(copy.deepcopy(pred_multi_ary[_idx - p_idx][sorted_idxs[_idx - p_idx][xy_idx]][n_idx]))

                        #         # # 信頼度が足りず、あり得ない値が設定されている場合、一つ前のをチェックする
                        #         # if openpose_x <= 0 and openpose_y <= 0:
                        #         #     # 関節位置を取得する
                        #         #     openpose_x = openpose_2d[_idx - 1][sorted_idxs[_idx - 1][xy_idx]][n_idx][0]
                        #         #     openpose_y = openpose_2d[_idx - 1][sorted_idxs[_idx - 1][xy_idx]][n_idx][1]

                        #         # logger.debug("pred_multi_ary: %s", pred_multi_ary)
                        #         logger.debug("_idx:%s, xy_idx: %s, n_idx: %s, sorted: %s, len(pred_multi_ary):%s", _idx, xy_idx, n_idx, sorted_idxs[_idx - 1][xy_idx], len(pred_multi_ary[_idx - 1][sorted_idxs[_idx - 1][xy_idx]]))

                        if openpose_x > 0 and openpose_y > 0:
                            # 信頼度が足りている場合、縮尺に応じた深度を取得

                            # オリジナルの画像サイズから、縮尺を取得
                            scale_orig_x = openpose_x / orig_width
                            scale_orig_y = openpose_y / orig_height

                            # logger.debug("s: %s, n: %s, scale_orig_x: %s, scale_orig_y: %s", _idx, n_idx, scale_orig_x, scale_orig_y)

                            # 縮尺を展開して、深度解析後の画像サイズに合わせる
                            pred_x = int(pred_width * scale_orig_x)
                            pred_y = int(pred_height * scale_orig_y)
                            # logger.debug("s: %s, n: %s, pred_x: %s, pred_y: %s, len(pred[0]): %s", _idx, n_idx, pred_x, pred_y, len(pred))

                            if 0 <= pred_y < len(pred[0]) and 0 <= pred_x < len(pred[0][pred_y]):
                                # たまにデータが壊れていて、「9.62965e-35」と取れてしまった場合の対策
                                depth = pred[0][pred_y][pred_x][0]
                            else:
                                pred_ary.append(0)

                            # logger.debug("s: %s, n: %s, pred_x: %s, pred_y: %s, depth: %s", _idx, n_idx, pred_x, pred_y, depth)

                            pred_ary.append(depth)
                            pred_xy.append([pred_x, pred_y])
                        else:
                            # 信頼度がどうしても足りない場合、0設定
                            pred_ary.append(0)

                    # logger.debug("pred_ary: %s", pred_ary)
                    pred_multi_ary[_idx][xy_idx] = pred_ary
                    # logger.debug("pred_multi_ary: %s", pred_multi_ary)
                    pred_multi_xys[xy_idx] = pred_xy

                    if xy_idx == 0 and _idx == 0:
                        # 1人目の0F目の場合、基準深度として平均値を保存
                        # 深度0が含まれていると狂うので、ループしてチェックしつつ合算
                        pred_sum = 0
                        pred_cnt = 0
                        for pred_one in pred_ary:
                            if pred_one > 0:
                                pred_sum += pred_one
                                pred_cnt += 1

                        base_depth = pred_sum / pred_cnt

                        logger.debug(
                            "基準深度取得: base_depth: %s, pred_sum: %s, pred_cnt: %s", base_depth, pred_sum, pred_cnt)

                # # 0F目は算出後の深度を改めて減算
                # if _idx == 0:
                #     for pred_one_ary in pred_multi_ary[_idx]:
                #         for pred_one_idx, pred_one in enumerate(pred_one_ary):
                #             if pred_one > 0:
                #                 pred_one_ary[pred_one_idx] = pred_one_ary[pred_one_idx] - base_depth

                #     logger.debug("0F目深度: %s", pred_multi_ary[_idx])

                # 深度画像保存 -----------------------
                if level[verbose] <= logging.INFO:
                    # Plot result
                    plt.cla()
                    plt.clf()
                    ii = plt.imshow(pred[0, :, :, 0], interpolation='nearest')
                    plt.colorbar(ii)

                    # 散布図のようにして、出力に使ったポイントを明示
                    for pred_xy, color in zip(pred_multi_xys, ["#33FF33", "#3333FF", "#FFFFFF", "#FFFF33", "#FF33FF", "#33FFFF", "#00FF00", "#0000FF", "#666666", "#FFFF00", "#FF00FF", "#00FFFF"]):
                        for xy in pred_xy:
                            plt.scatter(xy[0], xy[1], s=5, c=color)

                    plotName = "{0}/depth_{1:012d}.png".format(subdir, cnt)
                    plt.savefig(plotName)
                    logger.debug("Save: {0}".format(plotName))

                    # アニメーションGIF用に保持
                    png_lib.append(imageio.imread(plotName))

                    plt.close()

            # 人体別処理用インデックス（深度補間を有効にするため、一区切り前のを使用する）
            _iidx = _idx - interval

            if _iidx < 0:
                # まだ処理インデックスが開始してない場合、スキップ
                cnt += 1
                continue

            # JSONファイルを読み直す
            _file = os.path.join(openpose_output_dir,
                                 openpose_filenames[_iidx])
            if not os.path.isfile(_file): raise Exception(
                "No file found!!, {0}".format(_file))
            try:
                data = json.load(open(_file))
            except Exception as e:
                logger.warn("JSON読み込み失敗のため、空データ読み込み, %s %s", _file, e)
                data = json.load(
                    open("tensorflow/json/all_empty_keypoints.json"))

            for i in range(len(data["people"]), people_size):
                # 足りない分は空データを埋める
                data["people"].append(json.load(open("tensorflow/json/one_keypoints.json")))

            logger.info("人体別処理: idx: %s, iidx: %s file: %s ------",
                        _idx, _iidx, openpose_filenames[_iidx])

            # 人数を計算
            _len = len(data["people"])

            # logger.debug("pred_multi_ary: %s", pred_multi_ary)

            # インデックス並び替え -------------------------
            # 開始時
            if _iidx == 0:
                # 前回のXYを保持
                past_data = data["people"]

                # 最初は左から順番に0番目,1番目と並べる
                # FIXME 初期表示時の左から順番に並べたい
                # first_sorted_idxs = sort_first_idxs(data["people"])
                # logger.info("first_sorted_idxs: %s", first_sorted_idxs)

                for pidx in range(_len):
                    # 最初はインデックスの通りに並べる
                    sorted_idxs[0][pidx] = pidx

                past_depth_idx = -1
                next_depth_idx = -1
            else:
                if len(data["people"]) <= 0:
                    logger.info("今回データなしの為、前回ソート順流用 _iidx: %s(%s)",
                                _iidx, _display_idx)
                    sorted_idxs[_iidx], is_all_reverses[_iidx], is_upper_reverses[_iidx], is_lower_reverses[_iidx] = copy.deepcopy(
                        sorted_idxs[_iidx - 1]), copy.deepcopy(is_all_reverses[_iidx - 1]), copy.deepcopy(is_upper_reverses[_iidx]), copy.deepcopy(is_lower_reverses[_iidx])
                else:
                    # 片足寄せであるか計算（片寄せの場合、足の信頼度ゼロ）
                    if _iidx in reverse_frame_dict:
                        is_onesides[_iidx] = calc_oneside(
                            sorted_idxs[_iidx - 1], past_data, data["people"], True)

                    if _iidx in order_specific_dict:
                        # 順番指定リストに該当フレームがある場合
                        for key_idx, person_idx in enumerate(order_specific_dict[_iidx]):
                            # Openposeのデータの順番に応じたソート順を指定する
                            sorted_idxs[_iidx][key_idx] = person_idx
                            # 反転はさせない
                            is_all_reverses[_iidx][key_idx] = False
                            is_upper_reverses[_iidx][key_idx] = False
                            is_lower_reverses[_iidx][key_idx] = False
                            # logger.info("_iidx: %s, _display_idx: %s, key_idx: %s, person_idx: %s", _iidx, _display_idx, key_idx, person_idx )

                        file_logger.warn("※※{0:05d}F目 順番指定あり {1}".format(
                            _iidx, order_specific_dict[_iidx]))
                        # logger.info("_iidx: %s, _display_idx: %s, sorted_idxs[_iidx]: %s", _iidx, _display_idx, sorted_idxs[_iidx] )
                    else:
                        # 前回の深度
                        past_depth_idx = _iidx - (_iidx % interval)
                        # 次回の深度
                        next_depth_idx = _iidx + interval - (_iidx % interval)
                        if next_depth_idx >= len(openpose_2d):
                            # 最後は同じ値をnextとして見る
                            next_depth_idx = len(openpose_2d) - 1

                        # 前回のXYと深度から近いindexを算出
                        sorted_idxs[_iidx], is_all_reverses[_iidx], is_upper_reverses[_iidx], is_lower_reverses[_iidx] = calc_nearest_idxs(
                            sorted_idxs[_iidx - 1], past_data, data["people"], pred_multi_ary[past_depth_idx], pred_multi_ary[next_depth_idx])

            logger.info("＊＊_iidx: %s(%s), past_depth_idx: %s, next_depth_idx: %s, sorted_idxs: %s, all: %s, upper: %s, lower: %s", _iidx, _display_idx,
                        past_depth_idx, next_depth_idx, sorted_idxs[_iidx], is_all_reverses[_iidx], is_upper_reverses[_iidx], is_lower_reverses[_iidx])

            # JSONファイルを読み直す(片足寄せの場合、足の信頼度が消えているため)
            _file = os.path.join(openpose_output_dir, openpose_filenames[_iidx])
            if not os.path.isfile(_file): raise Exception("No file found!!, {0}".format(_file))
            try:
                data = json.load(open(_file))
                # 過去データ上書きしないデータも保持
                org_data = json.load(open(_file))
            except Exception as e:
                logger.warn("JSON読み込み失敗のため、空データ読み込み, %s %s", _file, e)
                data = json.load(open("tensorflow/json/all_empty_keypoints.json"))
                org_data = json.load(open("tensorflow/json/all_empty_keypoints.json"))

            # 現在データ
            now_data = [[] for x in range(people_size)]
            # 過去を引き継いだ現在データ
            all_now_data = [[] for x in range(people_size)]

            for i in range(len(data["people"]), people_size):
                # 足りない分は空データを埋める
                data["people"].append(json.load(open("tensorflow/json/one_keypoints.json")))
                org_data["people"].append(json.load(open("tensorflow/json/one_keypoints.json")))

            # インデックス出力 ------------------------------
            if _len <= 0:
                logger.debug("空データ出力 %s %s", _iidx, _len)
                # まったくデータがない場合、空データを投入する
                for pidx, sidx in enumerate(sorted_idxs[_iidx]):
                    # 一旦空データを読む
                    outputdata = json.load(
                        open("tensorflow/json/empty_keypoints.json"))

                    # インデックス対応分のディレクトリ作成
                    idx_path = '{0}/{1}_{3}_idx{2:02d}/json/{4}'.format(os.path.dirname(
                        openpose_output_dir), os.path.basename(openpose_output_dir), pidx+1, now_str, openpose_filenames[_iidx])
                    os.makedirs(os.path.dirname(idx_path), exist_ok=True)

                    # 出力
                    # json.dump(data, open(idx_path,'w'), indent=4)
                    json.dump(outputdata, open(idx_path, 'w'))

                    if _iidx == 0 or _iidx % interval == 0:
                        # 深度データ
                        depth_idx_path = '{0}/{1}_{3}_idx{2:02d}/depth.txt'.format(os.path.dirname(
                            openpose_output_dir), os.path.basename(openpose_output_dir), pidx+1, now_str)
                        # 追記モードで開く
                        depthf = open(depth_idx_path, 'a')
                        pred_ary = [str(0) for x in range(18)]
                        # 一行分を追記
                        depthf.write("{0}, {1}\n".format(
                            _iidx, ','.join(pred_ary)))
                        depthf.close()

                    now_data[sidx] = outputdata
            else:
                if _iidx in reverse_frame_dict:
                    logger.debug("反転判定対象フレーム: %s(%s)", _iidx, _display_idx)

                # 何らかのデータがある場合
                for pidx, sidx in enumerate(sorted_idxs[_iidx]):
                    # 現在データ(sidxで振り分け済み)
                    now_sidx_data = data["people"][sidx]["pose_keypoints_2d"]

                    if _iidx > 0:
                        # とりあえず何らかのデータがある場合
                        # 過去データ
                        past_pidx_data = past_data[pidx]["pose_keypoints_2d"]

                        for o in range(0,len(now_sidx_data),3):
                            oidx = int(o/3)
                            if now_sidx_data[o] == now_sidx_data[o+1] == 0 and oidx in [1,2,3,4,5,6,7,8,9,10,11,12,13]:
                                logger.debug("過去PU: pidx: %s, sidx:%s, o: %s, ns: %s, pp: %s, np: %s, ps: %s", pidx, sidx, oidx, now_sidx_data[o], past_pidx_data[o], data["people"][pidx]["pose_keypoints_2d"][o], past_data[sidx]["pose_keypoints_2d"][o])
                                logger.debug("sidx: %s, now_sidx_data: %s", sidx, now_sidx_data)
                                # XもYも0の場合、過去から引っ張ってくる
                                # is_now_upper_reversedはまだ判定してないので前フレームの反転結果
                                if is_now_upper_reversed[pidx] and is_now_lower_reversed[pidx]:
                                    # 反転している場合、反転INDEX(全身)
                                    now_sidx_data[o] = past_pidx_data[OPENPOSE_REVERSE_ALL[oidx]*3]
                                    now_sidx_data[o+1] = past_pidx_data[OPENPOSE_REVERSE_ALL[oidx]*3+1]
                                    now_sidx_data[o+2] = past_pidx_data[OPENPOSE_REVERSE_ALL[oidx]*3+2]
                                    # logger.debug("全反: o: %s, revo: %s, org: %s, rev: %s", o, OPENPOSE_REVERSE_ALL[int(o/3)]*3, past_pidx_data[o], past_pidx_data[OPENPOSE_REVERSE_ALL[int(o/3)]*3])
                                elif is_now_upper_reversed[pidx] and is_now_lower_reversed[pidx] == False :
                                    # logger.debug("反: %s", data["people"][sidx]["pose_keypoints_2d"][OPENPOSE_REVERSE_ALL[int(o/3)]*3])
                                    # 反転している場合、反転INDEX(上半身)
                                    now_sidx_data[o] = past_pidx_data[OPENPOSE_REVERSE_UPPER[oidx]*3]
                                    now_sidx_data[o+1] = past_pidx_data[OPENPOSE_REVERSE_UPPER[oidx]*3+1]
                                    now_sidx_data[o+2] = past_pidx_data[OPENPOSE_REVERSE_UPPER[oidx]*3+2]
                                    # logger.debug("上反: o: %s, revo: %s, org: %s, rev: %s", o, OPENPOSE_REVERSE_UPPER[int(o/3)]*3, past_pidx_data[o], past_pidx_data[OPENPOSE_REVERSE_UPPER[int(o/3)]*3])
                                elif is_now_upper_reversed[pidx] == False and is_now_lower_reversed[pidx]:
                                    # logger.debug("反: %s", data["people"][sidx]["pose_keypoints_2d"][OPENPOSE_REVERSE_ALL[int(o/3)]*3])
                                    # 反転している場合、反転INDEX(下半身)
                                    now_sidx_data[o] = past_pidx_data[OPENPOSE_REVERSE_LOWER[oidx]*3]
                                    now_sidx_data[o+1] = past_pidx_data[OPENPOSE_REVERSE_LOWER[oidx]*3+1]
                                    now_sidx_data[o+2] = past_pidx_data[OPENPOSE_REVERSE_LOWER[oidx]*3+2]
                                    # logger.debug("下反: o: %s, revo: %s, org: %s, rev: %s", o, OPENPOSE_REVERSE_LOWER[int(o/3)]*3, past_pidx_data[o], past_pidx_data[OPENPOSE_REVERSE_LOWER[int(o/3)]*3])
                                else:
                                    # logger.debug("正: %s", data["people"][sidx]["pose_keypoints_2d"][o])
                                    now_sidx_data[o] = past_pidx_data[o]
                                    now_sidx_data[o+1] = past_pidx_data[o+1]
                                    now_sidx_data[o+2] = past_pidx_data[o+2]
                                                            
                                # data["people"][sidx]["pose_keypoints_2d"][o] = past_data[pidx]["pose_keypoints_2d"][o]
                                # data["people"][sidx]["pose_keypoints_2d"][o+1] = past_data[pidx]["pose_keypoints_2d"][o+1]
                                # data["people"][sidx]["pose_keypoints_2d"][o+2] = past_data[pidx]["pose_keypoints_2d"][o+2]

                    if _iidx > 0 and _iidx in reverse_frame_dict:
                        # 前回のXYと深度から近いindexを算出
                        # 埋まってない部分を補完して、改めて反転再算出
                        # 既に並べ終わってるので、少し上げ底して厳しめにチェックする
                        _, is_retake_all_reverses, is_retake_upper_reverses, is_retake_lower_reverses = \
                            calc_nearest_idxs([0], [past_data[pidx]], [data["people"][sidx]], [pred_multi_ary[past_depth_idx][sidx]], [pred_multi_ary[next_depth_idx][sidx]], 0.03)
                        
                        logger.debug("反転再チェック: _iidx: %s, pidx: %s, all: %s, upper: %s, lower: %s", _iidx, pidx, is_retake_all_reverses[0], is_retake_upper_reverses[0], is_retake_lower_reverses[0])

                        is_all_reverses[_iidx][pidx] = is_retake_all_reverses[0]
                        is_upper_reverses[_iidx][pidx] = is_retake_upper_reverses[0]
                        is_lower_reverses[_iidx][pidx] = is_retake_lower_reverses[0]

                    # 出力対象となるpeople内のINDEX反転有無
                    is_all_reverse = is_all_reverses[_iidx][pidx]
                    # 上半身反転有無
                    is_upper_reverse = is_upper_reverses[_iidx][pidx]
                    # 下半身反転有無
                    is_lower_reverse = is_lower_reverses[_iidx][pidx]

                    if _iidx in reverse_frame_dict:

                        if _iidx > 0:
                            # 片寄せ有無(入れ替えた後なので、sidx参照)
                            if is_onesides[_iidx][sidx]:
                                # 片寄せの場合、前回を引き継ぐ
                                is_all_reverses[_iidx][pidx] = is_all_reverses[_iidx-1][pidx]
                                is_upper_reverses[_iidx][pidx] = is_upper_reverses[_iidx-1][pidx]
                                is_lower_reverses[_iidx][pidx] = is_lower_reverses[_iidx-1][pidx]

                        if is_upper_reverse:
                            # 出力対象が反転の場合、現在の状態から反転させる(上半身)
                            is_now_upper_reversed[pidx] = not(is_now_upper_reversed[pidx])

                        if is_lower_reverse:
                            # 出力対象が反転の場合、現在の状態から反転させる(下半身)
                            is_now_lower_reversed[pidx] = not(is_now_lower_reversed[pidx])
                        
                        if is_all_reverse:
                            # 全身反転の場合
                            if is_now_upper_reversed[pidx] != is_now_lower_reversed[pidx]:
                                # 上半身と下半身で反転が違う場合、反転クリア
                                is_now_upper_reversed[pidx] = False
                                is_now_lower_reversed[pidx] = False
                            else:
                                # 反転状況が同じ場合は、そのまま反転
                                is_now_upper_reversed[pidx] = not(is_now_upper_reversed[pidx])
                                is_now_lower_reversed[pidx] = not(is_now_lower_reversed[pidx])
                    else:
                        # 反転対象外の場合、クリア
                        is_now_upper_reversed[pidx] = False
                        is_now_lower_reversed[pidx] = False
                        
                    logger.debug("pidx: %s, is_now_upper_reversed: %s, is_now_lower_reversed: %s", pidx, is_now_upper_reversed[pidx], is_now_lower_reversed[pidx])

                    # # トレース失敗の場合、クリア
                    # if (is_all_reverse == False and (is_upper_reverse or (is_upper_reverse == False and is_now_upper_reversed[pidx] ))) and (targetdata[2*3] == 0 or targetdata[3*3] == 0 or targetdata[5*3] == 0 or targetdata[6*3] == 0) :
                    #     logger.debug("上半身ひじまでのトレース失敗のため、上半身反転フラグクリア %s(%s) data: %s", _iidx, _display_idx, targetdata)
                    #     is_upper_reverses[_iidx][pidx] = False
                    #     is_now_upper_reversed[pidx] = False

                    # if (is_all_reverse == False or (is_lower_reverse or (is_lower_reverse == False and is_now_lower_reversed[pidx] ))) and (targetdata[8*3] == 0 or targetdata[9*3] == 0 or targetdata[11*3] == 0 or targetdata[12*3] == 0) :
                    #     logger.debug("下半身ひざまでのトレース失敗のため、下半身反転フラグクリア %s(%s) data: %s", _iidx, _display_idx, targetdata)
                    #     is_lower_reverses[_iidx][pidx] = False
                    #     is_now_lower_reversed[pidx] = False

                    logger.debug("is_now_upper_reversed: %s, is_now_lower_reversed: %s", is_now_upper_reversed, is_now_lower_reversed)

                    logger.debug("_iidx: %s(%s), sidx: %s, pidx: %s, upper: %s, lower: %s", _iidx, _display_idx, sidx, pidx, is_now_upper_reversed[pidx], is_now_lower_reversed[pidx])

                    if _iidx in reverse_frame_dict:
                        if is_now_upper_reversed[pidx] and is_now_lower_reversed[pidx]:
                            file_logger.warn("※※{0:05d}F目 {1}番目の人物、全身反転 [{2}:{3}]".format( _iidx, pidx, _iidx, pidx))
                        elif is_now_upper_reversed[pidx] and is_now_lower_reversed[pidx] == False :
                            file_logger.warn("※※{0:05d}F目 {1}番目の人物、上半身反転 [{2}:{3}]".format( _iidx, pidx, _iidx, pidx))
                        elif is_now_upper_reversed[pidx] == False and is_now_lower_reversed[pidx]:
                            file_logger.warn("※※{0:05d}F目 {1}番目の人物、下半身反転 [{2}:{3}]".format( _iidx, pidx, _iidx, pidx))
                            
                    # 一旦空データを読む
                    outputdata = json.load(open("tensorflow/json/empty_keypoints.json"))
                    # 一旦空データを読む
                    all_outputdata = json.load(open("tensorflow/json/empty_keypoints.json"))

                    # 過去の上書きがない元データ
                    org_sidx_data = org_data["people"][sidx]["pose_keypoints_2d"]

                    for o in range(0,len(outputdata["people"][0]["pose_keypoints_2d"]),3):
                        # デフォルトのXINDEX
                        oidx = int(o/3)

                        if is_now_upper_reversed[pidx] and is_now_lower_reversed[pidx]:
                            oidx = OPENPOSE_REVERSE_ALL[oidx]
                        elif is_now_upper_reversed[pidx] and is_now_lower_reversed[pidx] == False :
                            # 反転している場合、反転INDEX(上半身)
                            oidx = OPENPOSE_REVERSE_UPPER[oidx]
                        elif is_now_upper_reversed[pidx] == False and is_now_lower_reversed[pidx]:
                            # 反転している場合、反転INDEX(下半身)
                            oidx = OPENPOSE_REVERSE_LOWER[oidx]
                        
                        if oidx in [1,2,5,8,11]:
                            # 体幹は、過去データ込みでコピー
                            outputdata["people"][0]["pose_keypoints_2d"][o] = now_sidx_data[oidx*3]
                            outputdata["people"][0]["pose_keypoints_2d"][o+1] = now_sidx_data[oidx*3+1]
                            outputdata["people"][0]["pose_keypoints_2d"][o+2] = now_sidx_data[oidx*3+2]
                        else:
                            # その他はオリジナルデータのみコピー
                            outputdata["people"][0]["pose_keypoints_2d"][o] = org_sidx_data[oidx*3]
                            outputdata["people"][0]["pose_keypoints_2d"][o+1] = org_sidx_data[oidx*3+1]
                            outputdata["people"][0]["pose_keypoints_2d"][o+2] = org_sidx_data[oidx*3+2]

                        # 過去引継データは、全部過去込みで引き継ぐ
                        all_outputdata["people"][0]["pose_keypoints_2d"][o] = now_sidx_data[oidx*3]
                        all_outputdata["people"][0]["pose_keypoints_2d"][o+1] = now_sidx_data[oidx*3+1]
                        all_outputdata["people"][0]["pose_keypoints_2d"][o+2] = now_sidx_data[oidx*3+2]

                        if outputdata["people"][0]["pose_keypoints_2d"][o] > orig_width or outputdata["people"][0]["pose_keypoints_2d"][o] < 0 \
                            or outputdata["people"][0]["pose_keypoints_2d"][o+1] > orig_height or outputdata["people"][0]["pose_keypoints_2d"][o+1] < 0 :
                            # 画像範囲外のデータが取れた場合、とりあえず0を入れ直す
                            outputdata["people"][0]["pose_keypoints_2d"][o] = 0
                            outputdata["people"][0]["pose_keypoints_2d"][o+1] = 0
                            outputdata["people"][0]["pose_keypoints_2d"][o+2] = 0

                    if _iidx > 0 and _iidx in reverse_frame_dict:
                        # 出力データでもう一度片足寄せであるか計算
                        is_result_oneside = calc_oneside([0], [past_data[pidx]], outputdata["people"])

                        if True in is_result_oneside:
                            # 片寄せの可能性がある場合、前回データをコピー
                            file_logger.warn("※※{0:05d}F目 {1}番目の人物、片寄せ可能性あり".format( _iidx, pidx))

                            for _lidx, _lval in enumerate([8,9,10,11,12,13]):
                                outputdata["people"][0]["pose_keypoints_2d"][_lval*3] = past_pidx_data[_lval*3]
                                outputdata["people"][0]["pose_keypoints_2d"][_lval*3+1] = past_pidx_data[_lval*3+1]
                                # 信頼度は半分
                                conf = past_pidx_data[_lval*3+2]/2
                                outputdata["people"][0]["pose_keypoints_2d"][o+2] = conf if 0 < conf < 1 else 0.3
                            
                    logger.debug("outputdata %s", outputdata["people"][0]["pose_keypoints_2d"])

                    # 出力順番順に並べなおしてリストに設定
                    now_data[sidx] = outputdata
                    all_now_data[sidx] = all_outputdata

            # 首の位置が一番よく取れてるので、首の位置を出力する
            display_nose_pos = {}
            for pidx, sidx in enumerate(sorted_idxs[_iidx]):   
                # データがある場合、そのデータ
                display_nose_pos[sidx] = [now_data[pidx]["people"][0]["pose_keypoints_2d"][1*3], now_data[pidx]["people"][0]["pose_keypoints_2d"][1*3+1]]

                # インデックス対応分のディレクトリ作成
                idx_path = '{0}/{1}_{3}_idx{2:02d}/json/{4}'.format(os.path.dirname(openpose_output_dir), os.path.basename(openpose_output_dir), sidx+1, now_str, openpose_filenames[_iidx])
                os.makedirs(os.path.dirname(idx_path), exist_ok=True)
                
                # 出力
                # json.dump(data, open(idx_path,'w'), indent=4)
                json.dump(now_data[pidx], open(idx_path,'w'), indent=4)

                if _iidx % interval == 0:
                    # 深度データ
                    depth_idx_path = '{0}/{1}_{3}_idx{2:02d}/depth.txt'.format(os.path.dirname(openpose_output_dir), os.path.basename(openpose_output_dir), pidx+1, now_str)
                    # 追記モードで開く
                    depthf = open(depth_idx_path, 'a')
                    # 深度データを文字列化する
                    # logger.debug("pred_multi_ary[_idx]: %s", pred_multi_ary[_idx])
                    # logger.debug("pred_multi_ary[_idx][sidx]: %s", pred_multi_ary[_idx][sidx])
                    pred_str_ary = []
                    if is_now_upper_reversed[pidx] and is_now_lower_reversed[pidx]:
                        # 反転している場合、反転INDEX
                        for _oidx, o in enumerate(pred_multi_ary[_iidx][sidx]):
                            pred_multi_ary[_iidx][sidx][_oidx] = pred_multi_ary[_iidx][sidx][OPENPOSE_REVERSE_ALL[_oidx]]
                    elif is_now_upper_reversed[pidx] and is_now_lower_reversed[pidx] == False:
                        # 反転している場合、反転INDEX(上半身)
                        for _oidx, o in enumerate(pred_multi_ary[_iidx][sidx]):
                            pred_multi_ary[_iidx][sidx][_oidx] = pred_multi_ary[_iidx][sidx][OPENPOSE_REVERSE_UPPER[_oidx]]
                    elif is_now_upper_reversed[pidx] == False and is_now_lower_reversed[pidx]:
                        # 反転している場合、反転INDEX(上半身)
                        for _oidx, o in enumerate(pred_multi_ary[_iidx][sidx]):
                            pred_multi_ary[_iidx][sidx][_oidx] = pred_multi_ary[_iidx][sidx][OPENPOSE_REVERSE_LOWER[_oidx]]

                    pred_str_ary = [ str(x - base_depth) for x in pred_multi_ary[_iidx][sidx] ]

                    # 一行分を追記
                    depthf.write("{0}, {1}\n".format(_display_idx, ','.join(pred_str_ary)))
                    depthf.close()

            file_logger.warn("＊＊{0:05d}F目の出力順番: [{1}:{2}], 位置: {3}".format(_iidx, _iidx, ','.join(map(str, sorted_idxs[_iidx])), sorted(display_nose_pos.items()) ))

            if _iidx > 0:
                if len(data["people"]) <= 0:
                    # まったくデータが無い場合、前々回のデータをそのまま流用する(上書きしない)
                    logger.debug("過去データなしの為、前々回データ流用")
                    pass
                else:
                    # 出力し終わったら、過去引継データを過去データとして保持する
                    for pidx, sidx in enumerate(sorted_idxs[_iidx]):
                        past_data[sidx] = all_now_data[pidx]["people"][0]

            # インクリメント        
            cnt += 1
        
            # if _iidx >= 4050:
            #     break

    if level[verbose] <= logging.INFO:
        logger.info("creating Gif {0}/movie_depth.gif, please Wait!".format(os.path.dirname(depth_path)))
        imageio.mimsave('{0}/movie_depth.gif'.format(os.path.dirname(depth_path)), png_lib, fps=30)

    # 終わったら後処理
    cap.release()
    cv2.destroyAllWindows()

    if is_avi_output:

        fourcc_names = ["I420"]

        if os.name == "nt":
            # Windows
            fourcc_names = ["IYUV"]

        # MMD用AVI出力 -----------------------------------------------------
        for fourcc_name in fourcc_names:
            try:
                # コーデックは実行環境によるので、自環境のMMDで確認できたfourccを総当たり
                # FIXME IYUVはAVI2なので、1GBしか読み込めない。ULRGは出力がULY0になってMMDで動かない。とりあえずIYUVを1GB以内で出力する
                fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
                # 出力先AVIを設定する（MMD用に小さめ)
                out_path = '{0}/output_{1}.avi'.format(depth_path, fourcc_name)

                avi_width = int(width*0.7)
                avi_height = int(height*0.7)

                out = cv2.VideoWriter(out_path, fourcc, 30.0, (avi_width, avi_height))
                
                if os.name == "nt":
                    # Windows
                    op_avi_path = re.sub(r'json$', "openpose.avi", openpose_output_dir)
                else:
                    op_avi_path = re.sub(r'json', "openpose.avi", openpose_output_dir)
                logger.warn("op_avi_path: %s", op_avi_path)
                # Openopse結果AVIを読み込む
                cnt = 0
                cap = cv2.VideoCapture(op_avi_path)

                while(cap.isOpened()):
                    # 動画から1枚キャプチャして読み込む
                    flag, frame = cap.read()  # Capture frame-by-frame

                    # 動画が終わっていたら終了
                    if flag == False:
                        break

                    for pidx, lcolor, rcolor in zip(range(people_size) \
                            , [(51,255,51), (255,51,51), (255,255,255), (51,255,255), (255,51,255), (255,255,51), (0,255,0), (255,0,0), (102,102,102), (0,255,255), (255,0,255), (255,255,0)] \
                            , [(51,51,255), (51,51,255),   (51,51,255),  (51,51,255),  (51,51,255),  (51,51,255), (0,0,255), (0,0,255),     (0,0,255),   (0,0,255),   (0,0,255),   (0,0,255)]):
                        # 人物別に色を設定, colorはBGR形式
                        # 【00番目】 左:緑, 右: 赤
                        # 【01番目】 左:青, 右: 赤
                        # 【02番目】 左:白, 右: 赤
                        # 【03番目】 左:黄, 右: 赤
                        # 【04番目】 左:桃, 右: 赤
                        # 【05番目】 左:濃緑, 右: 赤
                        # 【06番目】 左:濃青, 右: 赤
                        # 【07番目】 左:灰色, 右: 赤
                        # 【08番目】 左:濃黄, 右: 赤
                        # 【09番目】 左:濃桃, 右: 赤
                        idx_json_path = '{0}/{1}_{3}_idx{2:02d}/json/{4}'.format(os.path.dirname(openpose_output_dir), os.path.basename(openpose_output_dir), pidx+1, now_str, re.sub(r'\d{12}', "{0:012d}".format(cnt + start_frame), first_file_name))
                        # logger.warn("pidx: %s, color: %s, idx_json_path: %s", pidx, color, idx_json_path)

                        if os.path.isfile(idx_json_path):
                            data = json.load(open(idx_json_path))

                            for o in range(0,len(data["people"][0]["pose_keypoints_2d"]),3):
                                # 左右で色を分ける
                                color = rcolor if int(o/3) in [2,3,4,8,9,10,14,16] else lcolor

                                if data["people"][0]["pose_keypoints_2d"][o+2] > 0:
                                    # 少しでも信頼度がある場合出力
                                    # logger.debug("x: %s, y: %s", data["people"][0]["pose_keypoints_2d"][o], data["people"][0]["pose_keypoints_2d"][o+1])
                                    # cv2.drawMarker( frame, (int(data["people"][0]["pose_keypoints_2d"][o]+5), int(data["people"][0]["pose_keypoints_2d"][o+1]+5)), color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=10)
                                    # 座標のXY位置に点を置く。原点が左上なので、ちょっとずらす
                                    cv2.circle( frame, (int(data["people"][0]["pose_keypoints_2d"][o]+1), int(data["people"][0]["pose_keypoints_2d"][o+1]+1)), 5, color, thickness=-1)
                    
                    # 縮小
                    output_frame = cv2.resize(frame, (avi_width, avi_height))

                    # 全人物が終わったら出力
                    out.write(output_frame)

                    # インクリメント
                    cnt += 1

                logger.info('MMD用AVI: {0}'.format(out_path))

                # 出力に成功したら終了
                # break
            except Exception as e:
                logger.warn("MMD用AVI出力失敗: %s, %s", fourcc_name, e)

            finally:
                # 終わったら開放
                cap.release()
                out.release()
                cv2.destroyAllWindows()

# 0F目を左から順番に並べた人物INDEXを取得する
def sort_first_idxs(now_datas):
    most_common_idxs = []
    th = 0.3

    # 最終的な左からのINDEX
    result_nearest_idxs = [-1 for x in range(len(now_datas))]

    # 比較対象INDEX(最初は0(左端)を起点とする)
    target_x = [ 0 for x in range(int(len(now_datas[0]["pose_keypoints_2d"]))) ]

    # 人数分チェック
    for _idx in range(len(now_datas)):
        now_nearest_idxs = []
        # 関節位置Xでチェック
        for o in range(0,len(now_datas[0]["pose_keypoints_2d"]),3):
            is_target = True
            x_datas = []
            for _pnidx in range(len(now_datas)):
                if _pnidx not in result_nearest_idxs:
                    # 人物のうち、まだ左から並べられていない人物だけチェック対象とする
    
                    x_data = now_datas[_pnidx]["pose_keypoints_2d"][o]
                    x_conf = now_datas[_pnidx]["pose_keypoints_2d"][o+2]

                    if x_conf > th and is_target:
                        # 信頼度が一定以上あって、これまでも追加されている場合、追加
                        x_datas.append(x_data)
                    else:
                        # 一度でも信頼度が満たない場合、チェック対象外
                        is_target = False
                else:
                    # 既に並べられている人物の場合、比較対象にならない値を設定する
                    x_datas.append(sys.maxsize)
            
            # logger.info("sort_first_idxs: _idx: %s, x_datas: %s, is_target: %s", _idx, x_datas, is_target)

            if is_target:
                # 最終的に対象のままである場合、ひとつ前の人物に近い方のINDEXを取得する
                now_nearest_idxs.append(get_nearest_idx(x_datas, target_x[o]))

        # logger.info("sort_first_idxs: _idx: %s, now_nearest_idxs: %s", _idx, now_nearest_idxs)

        if len(now_nearest_idxs) > 0:
            # チェック対象件数がある場合、最頻出INDEXをチェックする
            most_common_idxs = Counter(now_nearest_idxs).most_common()
            logger.debug("sort_first_idxs: _idx: %s, most_common_idxs: %s", _idx, most_common_idxs)
            # 最頻出INDEX
            result_nearest_idxs[_idx] = most_common_idxs[0][0]
            # 次の比較元として、再頻出INDEXの人物を対象とする
            target_x = now_datas[most_common_idxs[0][0]]["pose_keypoints_2d"]

    logger.debug("sort_first_idxs: result_nearest_idxs: %s", result_nearest_idxs)

    if -1 in result_nearest_idxs:
        # 不採用になって判定できなかったデータがある場合
        for _nidx, _nval in enumerate(result_nearest_idxs):
            if _nval == -1:
                # 該当値が-1(判定不可）の場合
                for _cidx in range(len(now_datas)):
                    logger.debug("_nidx: %s, _nval: %s, _cidx: %s, _cidx not in nearest_idxs: %s", _nidx, _nval, _cidx, _cidx not in result_nearest_idxs)
                    # INDEXを頭から順に見ていく（正0, 正1 ... 正n, 逆0, 逆1 ... 逆n)
                    if _cidx not in result_nearest_idxs:
                        # 該当INDEXがリストに無い場合、設定
                        result_nearest_idxs[_nidx] = _cidx
                        break

    return result_nearest_idxs

# 前回のXYから片足寄せであるか判断する
def calc_oneside(past_sorted_idxs, past_data, now_data, is_oneside_reset=False):
    # ひざと足首のペア
    LEG_IDXS = [[9,12],[10,13]]

    # 過去のX位置データ
    is_past_oneside = False
    for _pidx, _idx in enumerate(past_sorted_idxs):
        past_xyc = past_data[_idx]["pose_keypoints_2d"]

        for _lidx, _lvals in enumerate(LEG_IDXS):
            logger.debug("past _idx: %s, _lidx: %s, %sx: %s, %sx: %s, %sy: %s, %sy:%s", _idx, _lidx, _lvals[0], past_xyc[_lvals[0]*3], _lvals[1], past_xyc[_lvals[1]*3], _lvals[0], past_xyc[_lvals[0]*3+1], _lvals[1], past_xyc[_lvals[1]*3+1])
            
            if past_xyc[_lvals[0]*3] > 0 and past_xyc[_lvals[1]*3] > 0 and past_xyc[_lvals[0]*3+1] > 0 and past_xyc[_lvals[1]*3+1] > 0 \
                and abs(past_xyc[_lvals[0]*3] - past_xyc[_lvals[1]*3]) < 5 and abs(past_xyc[_lvals[0]*3+1] - past_xyc[_lvals[1]*3+1]) < 5:
                logger.debug("過去片寄せ: %s(%s), (%s,%s), (%s,%s)", _pidx, _lidx, past_xyc[_lvals[0]*3], past_xyc[_lvals[1]*3], past_xyc[_lvals[0]*3+1], past_xyc[_lvals[1]*3+1] )
                # 誰かの足が片寄せっぽいならば、FLG＝ON
                is_past_oneside = True

    is_onesides = [ False for x in range(len(now_data)) ]
    # 今回のX位置データ
    for _idx in range(len(now_data)):
        now_xyc = now_data[_idx]["pose_keypoints_2d"]

        is_now_oneside_cnt = 0
        for _lidx, _lvals in enumerate(LEG_IDXS):
            logger.debug("now _idx: %s, _lidx: %s, %sx: %s, %sx: %s, %sy: %s, %sy:%s", _idx, _lidx, _lvals[0], now_xyc[_lvals[0]*3], _lvals[1], now_xyc[_lvals[1]*3], _lvals[0], now_xyc[_lvals[0]*3+1], _lvals[1], now_xyc[_lvals[1]*3+1])

            if now_xyc[_lvals[0]*3] > 0 and now_xyc[_lvals[1]*3] > 0 and now_xyc[_lvals[0]*3+1] > 0 and now_xyc[_lvals[1]*3+1] > 0 \
                and abs(now_xyc[_lvals[0]*3] - now_xyc[_lvals[1]*3]) < 5 and abs(now_xyc[_lvals[0]*3+1] - now_xyc[_lvals[1]*3+1]) < 5:
                # 両ひざ、両足首のX位置、Y位置がほぼ同じである場合
                logger.debug("現在片寄せ: %s(%s), (%s,%s), (%s,%s)", _idx, _lidx, now_xyc[_lvals[0]*3], now_xyc[_lvals[1]*3], now_xyc[_lvals[0]*3+1], now_xyc[_lvals[1]*3+1] )
                is_now_oneside_cnt += 1
        
        if is_now_oneside_cnt > 0 and is_past_oneside == False:
            for _lidx, _lval in enumerate([8,9,10,11,12,13]):
                # フラグを立てる
                is_onesides[_idx] == True
                # リセットFLG＝ONの場合、足の位置を一旦全部クリア
                if is_oneside_reset:
                    now_xyc[_lval*3] = 0
                    now_xyc[_lval*3+1] = 0
                    now_xyc[_lval*3+2] = 0

    return is_onesides


# 左右反転させたINDEX
OPENPOSE_REVERSE_ALL = {
    0: 0,
    1: 1,
    2: 5,
    3: 6,
    4: 7,
    5: 2,
    6: 3,
    7: 4,
    8: 11,
    9: 12,
    10: 13,
    11: 8,
    12: 9,
    13: 10,
    14: 15,
    15: 14,
    16: 17,
    17: 16,
    18: 18
}

# 上半身のみ左右反転させたINDEX
OPENPOSE_REVERSE_UPPER = {
    0: 0,
    1: 1,
    2: 5,
    3: 6,
    4: 7,
    5: 2,
    6: 3,
    7: 4,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 15,
    15: 14,
    16: 17,
    17: 16,
    18: 18
}

# 下半身のみ左右反転させたINDEX
OPENPOSE_REVERSE_LOWER = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 11,
    9: 12,
    10: 13,
    11: 8,
    12: 9,
    13: 10,
    14: 14,
    15: 15,
    16: 16,
    17: 17,
    18: 18
}

# 前回のXYと深度から近いindexを算出
def calc_nearest_idxs(past_sorted_idxs, past_data, now_data, past_pred_ary, now_pred_ary, limit_correction=0.0):
    # logger.debug("past_data: %s", past_data)
    
    # 前回の人物データ(前回のソート順に対応させる)
    # 左右反転もチェックするので、2倍。
    past_x_ary = [[] for x in range(len(past_data) * 2)]
    past_y_ary = [[] for x in range(len(past_data) * 2)]
    past_conf_ary = [[] for x in range(len(past_data) * 2)]
    # 下半身だけ回転しているパターン用
    past_lower_x_ary = [[] for x in range(len(past_data) * 2)]
    past_lower_y_ary = [[] for x in range(len(past_data) * 2)]
    past_lower_conf_ary = [[] for x in range(len(past_data) * 2)]
    # 上半身だけ回転しているパターン用
    past_upper_x_ary = [[] for x in range(len(past_data) * 2)]
    past_upper_y_ary = [[] for x in range(len(past_data) * 2)]
    past_upper_conf_ary = [[] for x in range(len(past_data) * 2)]
    for _idx, _idxv in enumerate(past_sorted_idxs):
        # logger.debug("past_data[_idx]: %s", past_data[_idx])

        past_xyc = past_data[_idx]["pose_keypoints_2d"]

        # logger.debug("_idx: %s, past_xyc: %s", _idx, past_xyc)
        # 正データ
        for o in range(0,len(past_xyc),3):
            # logger.debug("_idx: %s, o: %s", _idx, o)
            # 全身反転用
            past_x_ary[_idx].append(past_xyc[o])
            past_y_ary[_idx].append(past_xyc[o+1])
            past_conf_ary[_idx].append(past_xyc[o+2])

            # 下半身反転用
            past_lower_x_ary[_idx].append(past_xyc[o])
            past_lower_y_ary[_idx].append(past_xyc[o+1])
            past_lower_conf_ary[_idx].append(past_xyc[o+2])

            # 上半身反転用
            past_upper_x_ary[_idx].append(past_xyc[o])
            past_upper_y_ary[_idx].append(past_xyc[o+1])
            past_upper_conf_ary[_idx].append(past_xyc[o+2])
        # 反転データ
        for o in range(0,len(past_xyc),3):
            # logger.debug("_idx: %s, o: %s", _idx, o)
            past_x_ary[_idx + len(now_data)].append(past_xyc[OPENPOSE_REVERSE_ALL[int(o/3)]*3])
            past_y_ary[_idx + len(now_data)].append(past_xyc[OPENPOSE_REVERSE_ALL[int(o/3)]*3+1])
            # 反転は信頼度を下げる
            past_conf_ary[_idx + len(now_data)].append(past_xyc[OPENPOSE_REVERSE_ALL[int(o/3)]*3+2] - 0.1)
        # 下半身反転データ
        for o in range(0,len(past_xyc),3):
            # logger.debug("_idx: %s, o: %s", _idx, o)
            past_lower_x_ary[_idx + len(now_data)].append(past_xyc[OPENPOSE_REVERSE_LOWER[int(o/3)]*3])
            past_lower_y_ary[_idx + len(now_data)].append(past_xyc[OPENPOSE_REVERSE_LOWER[int(o/3)]*3+1])
            # 反転は信頼度を下げる
            past_lower_conf_ary[_idx + len(now_data)].append(past_xyc[OPENPOSE_REVERSE_LOWER[int(o/3)]*3+2] - 0.1)
        # 上半身反転データ
        for o in range(0,len(past_xyc),3):
            # logger.debug("_idx: %s, o: %s", _idx, o)
            past_upper_x_ary[_idx + len(now_data)].append(past_xyc[OPENPOSE_REVERSE_UPPER[int(o/3)]*3])
            past_upper_y_ary[_idx + len(now_data)].append(past_xyc[OPENPOSE_REVERSE_UPPER[int(o/3)]*3+1])
            # 反転は信頼度を下げる
            past_upper_conf_ary[_idx + len(now_data)].append(past_xyc[OPENPOSE_REVERSE_UPPER[int(o/3)]*3+2] - 0.1)
    
    logger.info("past_x: %s", np.array(past_x_ary)[:,1])

    # logger.debug("past_x_ary: %s", past_x_ary)
    # logger.debug("past_y_ary: %s", past_y_ary)

    # 今回の人物データ
    # 全身左右反転もチェックするので、2倍。
    now_x_ary = [[] for x in range(len(now_data) * 2)]
    now_y_ary = [[] for x in range(len(now_data) * 2)]
    now_conf_ary = [[] for x in range(len(now_data) * 2)]
    # 下半身だけ回転しているパターン用
    now_lower_x_ary = [[] for x in range(len(now_data) * 2)]
    now_lower_y_ary = [[] for x in range(len(now_data) * 2)]
    now_lower_conf_ary = [[] for x in range(len(now_data) * 2)]
    # 上半身だけ回転しているパターン用
    now_upper_x_ary = [[] for x in range(len(now_data) * 2)]
    now_upper_y_ary = [[] for x in range(len(now_data) * 2)]
    now_upper_conf_ary = [[] for x in range(len(now_data) * 2)]
    for _idx in range(len(now_data)):
        now_xyc = now_data[_idx]["pose_keypoints_2d"]
        # logger.debug("_idx: %s, now_xyc: %s", _idx, now_xyc)
        # 正データ
        for o in range(0,len(now_xyc),3):
            # logger.debug("_idx: %s, o: %s", _idx, o)
            now_x_ary[_idx].append(now_xyc[o])
            now_y_ary[_idx].append(now_xyc[o+1])
            now_conf_ary[_idx].append(now_xyc[o+2])

            # 下半身反転用
            now_lower_x_ary[_idx].append(now_xyc[o])
            now_lower_y_ary[_idx].append(now_xyc[o+1])
            now_lower_conf_ary[_idx].append(now_xyc[o+2])

            # 上半身反転用
            now_upper_x_ary[_idx].append(now_xyc[o])
            now_upper_y_ary[_idx].append(now_xyc[o+1])
            now_upper_conf_ary[_idx].append(now_xyc[o+2])
        # 反転データ
        for o in range(0,len(now_xyc),3):
            # logger.debug("_idx: %s, rev_idx: %s, o: %s, len(now_x_ary): %s, len(now_xyc): %s, OPENPOSE_REVERSE_ALL[o]: %s", _idx, _idx + len(now_data), o, len(now_x_ary), len(now_xyc), OPENPOSE_REVERSE_ALL[int(o/3)])
            now_x_ary[_idx + len(now_data)].append(now_xyc[OPENPOSE_REVERSE_ALL[int(o/3)]*3])
            now_y_ary[_idx + len(now_data)].append(now_xyc[OPENPOSE_REVERSE_ALL[int(o/3)]*3+1])
            # 反転は信頼度をすこし下げる
            now_conf_ary[_idx + len(now_data)].append(now_xyc[OPENPOSE_REVERSE_ALL[int(o/3)]*3+2] - 0.1)
        # 下半身反転データ
        for o in range(0,len(now_xyc),3):
            # logger.debug("_idx: %s, o: %s", _idx, o)
            now_lower_x_ary[_idx + len(now_data)].append(now_xyc[OPENPOSE_REVERSE_LOWER[int(o/3)]*3])
            now_lower_y_ary[_idx + len(now_data)].append(now_xyc[OPENPOSE_REVERSE_LOWER[int(o/3)]*3+1])
            now_lower_conf_ary[_idx + len(now_data)].append(now_xyc[OPENPOSE_REVERSE_LOWER[int(o/3)]*3+2] - 0.1)
        # 上半身反転データ
        for o in range(0,len(now_xyc),3):
            # logger.debug("_idx: %s, o: %s", _idx, o)
            now_upper_x_ary[_idx + len(now_data)].append(now_xyc[OPENPOSE_REVERSE_UPPER[int(o/3)]*3])
            now_upper_y_ary[_idx + len(now_data)].append(now_xyc[OPENPOSE_REVERSE_UPPER[int(o/3)]*3+1])
            now_upper_conf_ary[_idx + len(now_data)].append(now_xyc[OPENPOSE_REVERSE_UPPER[int(o/3)]*3+2] - 0.1)

    logger.info("now_x: %s", np.array(now_x_ary)[:,1])

    # 過去の深度データ
    past_pred = []
    for _pidx, _idx in enumerate(past_sorted_idxs):
        past_pred.append(past_pred_ary[_idx])

    # logger.debug("past_pred: %s,", past_pred)
    # logger.debug("org_past_conf: %s,", org_past_conf)

    # 信頼度の高い順に人物インデックスを割り当てていく
    avg_conf_ary = []
    for con in now_conf_ary:
        # 体幹ほど重みをつけて平均値を求める
        avg_conf_ary.append(np.average(np.array(con), weights=[0.5,2.0,0.5,0.3,0.1,0.5,0.3,0.1,0.8,0.3,0.1,0.8,0.3,0.1,0.1,0.1,0.1,0.1]))
    
    # 信頼度の低い順のインデックス番号
    conf_idxs = np.argsort(avg_conf_ary)
    logger.debug("avg_conf_ary: %s", avg_conf_ary)
    logger.debug("conf_idxs: %s", conf_idxs)

    # # 信頼度の高い順に人物インデックスを割り当てていく
    # normal_avg_conf_ary = []
    # for con in now_conf_ary[0:len(now_data)]:
    #     # 体幹ほど重みをつけて平均値を求める
    #     normal_avg_conf_ary.append(np.average(np.array(con), weights=[0.5,0.8,0.5,0.3,0.1,0.5,0.3,0.1,0.8,0.3,0.1,0.8,0.3,0.1,0.1,0.1,0.1,0.1]))
    
    # # 信頼度の低い順のインデックス番号
    # normal_conf_idxs = np.argsort(normal_avg_conf_ary)

    # conf_idxs = [-1 for x in range(len(now_conf_ary))]
    # for _ncidx in range(len(normal_conf_idxs)):
    #     # 正データ
    #     conf_idxs[_ncidx] = normal_conf_idxs[_ncidx]+len(now_data)
    #     # 反転データ
    #     conf_idxs[_ncidx+len(now_data)] = normal_conf_idxs[_ncidx]

    # logger.debug("normal_avg_conf_ary: %s", normal_avg_conf_ary)
    # logger.debug("normal_conf_idxs: %s", normal_conf_idxs)
    # logger.debug("conf_idxs: %s", conf_idxs)

    nearest_idxs = [-1 for x in range(len(conf_idxs))]
    is_upper_reverses = [False for x in range(len(conf_idxs))]
    is_lower_reverses = [False for x in range(len(conf_idxs))]
    most_common_idxs = []

    # logger.debug("past_pred_ary: %s", past_pred_ary)
    # logger.debug("now_pred_ary: %s", now_pred_ary)
    # XY正の判定用
    XY_LIMIT = 0.73 + limit_correction
    # XY上半身・下半身のみ反転用。やや厳しめ
    REV_LIMIT = 0.83 + limit_correction
    # 深度判定用。甘め
    D_LIMIT = 0.61 + limit_correction
    # 信頼度の低い順の逆順(信頼度降順)に人物を当てはめていく
    cidx = len(conf_idxs) - 1
    cidxcnt = 0
    while cidx >= 0 and cidxcnt < len(conf_idxs):
        now_conf_idx = conf_idxs[cidx]
        now_x = now_x_ary[now_conf_idx]
        now_y = now_y_ary[now_conf_idx]
        now_conf = now_conf_ary[now_conf_idx]

        logger.debug("cidx: %s, now_conf_idx: %s, %s", cidx, now_conf_idx, now_x)

        # 過去データの当該関節で、現在データと最も近いINDEXのリストを生成
        now_nearest_idxs, most_common_idxs, is_y, is_top = calc_most_common_idxs(conf_idxs, now_x, now_y, now_conf, past_x_ary, past_y_ary, past_lower_conf_ary)

        sum_most_common_idxs, most_common_per, sum_top2_per, top_frame, second_frame = \
            get_most_common_frames(most_common_idxs, conf_idxs)
        
        logger.debug("len(now_nearest_idxs): %s, all_size: %s, per: %s", len(now_nearest_idxs), (len(now_x) + ( 0 if is_y == False else len(now_y) )), len(now_nearest_idxs) / (len(now_x) + ( 0 if is_y == False else len(now_y) )))
        if len(now_nearest_idxs) / (len(now_x) + ( 0 if is_y == False else len(now_y) )) < 0.25 :
            # そもそもの絶対件数が少ない場合、深度チェック
            logger.debug("深度データチェック cidx: %s, now_conf_idx: %s", cidx, now_conf_idx)
            # 深度データは反転保持していないので、半分にする
            now_depth = now_pred_ary[int(now_conf_idx % len(now_data))]

            depth_now_nearest_idxs, depth_most_common_idxs = calc_depth_most_common_idxs(conf_idxs, now_depth, now_conf, past_pred, past_conf_ary, now_nearest_idxs)

            sum_depth_most_common_idxs, depth_most_common_per, depth_sum_top2_per, depth_top_frame, depth_second_frame = \
                get_most_common_frames(depth_most_common_idxs, conf_idxs)
                
            logger.debug("depth_most_common_per: %s, most_common_per: %s", depth_most_common_per, most_common_per)
            
            if depth_most_common_per > D_LIMIT and depth_most_common_per > most_common_per:
                now_nearest_idxs = depth_now_nearest_idxs
                most_common_idxs = depth_most_common_idxs
                logger.debug("＊深度データ採用: depth_now_nearest_idxs: %s, depth_most_common_idxs: %s", depth_now_nearest_idxs, depth_most_common_idxs)

        elif most_common_idxs[0][1] / len(now_nearest_idxs) < XY_LIMIT:
            # チェック対象件数が３割未満、再頻出が8割未満のいずれかの場合、
            # 上半身と下半身で回転が違っている可能性あり。

            logger.debug("下半身反転データチェック cidx: %s, now_conf_idx: %s", cidx, now_conf_idx)
            # 下半身だけ反転しているデータで比較する
            now_lower_x = now_lower_x_ary[now_conf_idx]
            now_lower_y = now_lower_y_ary[now_conf_idx]
            now_lower_conf = now_lower_conf_ary[now_conf_idx]

            lower_now_nearest_idxs, lower_most_common_idxs, is_y, is_top = calc_most_common_idxs(conf_idxs, now_lower_x, now_lower_y, now_lower_conf, past_lower_x_ary, past_lower_y_ary, past_conf_ary, OPENPOSE_REVERSE_LOWER )

            sum_lower_most_common_idxs, lower_most_common_per, lower_sum_top2_per, lower_top_frame, lower_second_frame = \
                get_most_common_frames(lower_most_common_idxs, conf_idxs)
            logger.debug("lower_most_common_per: %s, most_common_per: %s", lower_most_common_per, most_common_per)

            if lower_most_common_per > REV_LIMIT and lower_most_common_per > most_common_per:
                # 下半身反転データも同じINDEXで、より精度が高い場合、採用
                if now_x[2] == 0 or now_x[3] == 0 or now_x[5] == 0 or now_x[6] == 0:
                    # 上半身がない場合、全身反転とする
                    now_nearest_idxs = []
                    for lnni in lower_now_nearest_idxs:
                        now_nearest_idxs.append(lnni + len(now_data))

                    most_common_idxs = Counter(now_nearest_idxs).most_common()

                    for c in range(len(conf_idxs)):
                        is_existed = False
                        for m, mci in enumerate(most_common_idxs):
                            if c == most_common_idxs[m][0]:
                                is_existed = True
                                break
                        
                        if is_existed == False:
                            # 存在しないインデックスだった場合、追加
                            most_common_idxs.append( (c, 0) )
                    logger.debug("＊下半身→全身反転データ採用: now_nearest_idxs: %s, most_common_idxs: %s", now_nearest_idxs, most_common_idxs)
                else:
                    now_nearest_idxs = lower_now_nearest_idxs
                    most_common_idxs = lower_most_common_idxs
                    is_lower_reverses[now_conf_idx] = True
                    logger.debug("＊下半身反転データ採用: lower_now_nearest_idxs: %s, lower_most_common_idxs: %s, is_lower_reverses: %s", lower_now_nearest_idxs, lower_most_common_idxs, is_lower_reverses)
            else:
                # 信頼度が最後のものはチェックしない
                # 精度が高くない場合、上半身反転データチェック
                logger.debug("上半身反転データチェック cidx: %s, now_conf_idx: %s", cidx, now_conf_idx)

                # 上半身だけ反転しているデータで比較する
                now_upper_x = now_upper_x_ary[now_conf_idx]
                now_upper_y = now_upper_y_ary[now_conf_idx]
                now_upper_conf = now_upper_conf_ary[now_conf_idx]

                upper_now_nearest_idxs, upper_most_common_idxs, is_y, is_top = calc_most_common_idxs(conf_idxs, now_upper_x, now_upper_y, now_upper_conf, past_upper_x_ary, past_upper_y_ary, past_upper_conf_ary, OPENPOSE_REVERSE_UPPER)

                sum_upper_most_common_idxs, upper_most_common_per, upper_sum_top2_per, upper_top_frame, upper_second_frame = \
                    get_most_common_frames(upper_most_common_idxs, conf_idxs)
                logger.debug("upper_most_common_per: %s, most_common_per: %s", upper_most_common_per, most_common_per)

                if upper_most_common_per > REV_LIMIT and upper_most_common_per > most_common_per:
                    # 上半身反転データも同じINDEXで、より精度が高い場合、採用
                    if now_x[8] == 0 or now_x[9] == 0 or now_x[11] == 0 or now_x[12] == 0:
                        # 下半身がない場合、全身反転とする
                        now_nearest_idxs = []
                        for unni in upper_now_nearest_idxs:
                            now_nearest_idxs.append(unni + len(now_data))

                        most_common_idxs = Counter(now_nearest_idxs).most_common()

                        for c in range(len(conf_idxs)):
                            is_existed = False
                            for m, mci in enumerate(most_common_idxs):
                                if c == most_common_idxs[m][0]:
                                    is_existed = True
                                    break
                            
                            if is_existed == False:
                                # 存在しないインデックスだった場合、追加
                                most_common_idxs.append( (c, 0) )

                        logger.debug("＊上半身→全身反転データ採用: now_nearest_idxs: %s, most_common_idxs: %s", now_nearest_idxs, most_common_idxs)
                    else:
                        now_nearest_idxs = upper_now_nearest_idxs
                        most_common_idxs = upper_most_common_idxs
                        is_upper_reverses[now_conf_idx] = True
                        logger.debug("＊上半身反転データ採用: upper_now_nearest_idxs: %s, upper_most_common_idxs: %s, is_upper_reverses: %s", upper_now_nearest_idxs, upper_most_common_idxs, is_upper_reverses)
                else:
                    logger.debug("most_common_idxs: %s, lower_most_common_idxs: %s, upper_most_common_idxs: %s", most_common_idxs, lower_most_common_idxs, upper_most_common_idxs )

                    # TOP1もしくはTOP1.2で7.5割か
                    is_top = most_common_idxs[0][1] > len(now_nearest_idxs) * 0.71 or (sum_top2_per > 0.71 and top_frame == second_frame)
                    logger.debug("再検査:: sum_top2_per: %s, len(now_x): %s, top: %s, second: %s, is_top: %s", sum_top2_per, int(len(conf_idxs)/2), top_frame, second_frame, is_top)

                    if is_top :
                        logger.debug("全身TOP2の最頻出同一枠のため全身採用: sum_top2_per: %s, top: %s, second: %s", sum_top2_per, most_common_idxs[1][0] % len(now_data), most_common_idxs[1][0] % len(now_data))
                        is_upper_reverses[now_conf_idx] = False
                        is_lower_reverses[now_conf_idx] = False
                    else:
                        # 下半身反転も上半身反転もダメな場合、改めて深度チェック
                        logger.debug("深度データチェック cidx: %s, now_conf_idx: %s", cidx, now_conf_idx)
                        # 深度データは反転保持していないので、半分にする
                        now_depth = now_pred_ary[int(now_conf_idx % len(now_data))]

                        depth_now_nearest_idxs, depth_most_common_idxs = calc_depth_most_common_idxs(conf_idxs, now_depth, now_conf, past_pred, past_conf_ary, now_nearest_idxs)

                        sum_depth_most_common_idxs = 0
                        for lmci_data in depth_most_common_idxs:
                            sum_depth_most_common_idxs += lmci_data[1]

                        sum_most_common_idxs = 0
                        for mci_data in most_common_idxs:
                            sum_most_common_idxs += mci_data[1]

                        depth_most_common_per = 0 if sum_depth_most_common_idxs == 0 else depth_most_common_idxs[0][1] / sum_depth_most_common_idxs
                        most_common_per = 0 if sum_most_common_idxs == 0 else most_common_idxs[0][1] / sum_most_common_idxs
                        logger.debug("depth_most_common_per: %s, most_common_per: %s", depth_most_common_per, most_common_per)
                        
                        # depth_most_common_perの下限は甘め
                        if depth_most_common_per > D_LIMIT and depth_most_common_per > most_common_per:
                            now_nearest_idxs = depth_now_nearest_idxs
                            most_common_idxs = depth_most_common_idxs
                            is_upper_reverses[now_conf_idx] = False
                            is_lower_reverses[now_conf_idx] = False
                            logger.debug("＊深度データ採用: depth_now_nearest_idxs: %s, depth_most_common_idxs: %s", depth_now_nearest_idxs, depth_most_common_idxs)
                        else:
                            # 深度データも駄目だったので、とりあえずこれまでの中でもっとも確率の高いのを採用する
                            if most_common_idxs[0][0] in nearest_idxs and lower_most_common_per > most_common_per:
                                now_nearest_idxs = lower_now_nearest_idxs
                                most_common_idxs = lower_most_common_idxs
                                is_lower_reverses[now_conf_idx] = True
                                logger.debug("＊深度データ不採用→下半身反転データ採用: lower_now_nearest_idxs: %s, lower_most_common_idxs: %s, is_lower_reverses: %s", lower_now_nearest_idxs, lower_most_common_idxs, is_lower_reverses)
                            elif most_common_idxs[0][0] in nearest_idxs and upper_most_common_per > most_common_per:
                                now_nearest_idxs = upper_now_nearest_idxs
                                most_common_idxs = upper_most_common_idxs
                                is_upper_reverses[now_conf_idx] = True
                                logger.debug("＊深度データ不採用→上半身反転データ採用: upper_now_nearest_idxs: %s, upper_most_common_idxs: %s, is_upper_reverses: %s", upper_now_nearest_idxs, upper_most_common_idxs, is_upper_reverses)
                            else:
                                logger.debug("＊深度データ不採用→全身データ採用: upper_now_nearest_idxs: %s, upper_most_common_idxs: %s, is_upper_reverses: %s", upper_now_nearest_idxs, upper_most_common_idxs, is_upper_reverses)

        logger.debug("cidx: %s, most_common_idx: %s", cidx, most_common_idxs)
        
        is_passed = False
        # 最も多くヒットしたINDEXを処理対象とする
        for cmn_idx in range(len(most_common_idxs)):
            # 入れようとしているINDEXが、採用枠（前半）か不採用枠（後半）か
            if now_conf_idx < len(now_data):
                # 採用枠(前半)の場合
                check_ary = nearest_idxs[0: len(now_data)]
            else:
                # 不採用枠(後半)の場合
                check_ary = nearest_idxs[len(now_data): len(now_data)*2]
            
            logger.debug("nearest_idxs: %s, most_common_idxs[cmn_idx][0]: %s, check_ary: %s", nearest_idxs, most_common_idxs[cmn_idx][0], check_ary )

            is_idx_existed = False
            for ca in check_ary:
                logger.debug("ca: %s, ca / len(now): %s, most / len(now): %s", ca, ca % len(now_data), most_common_idxs[cmn_idx][0] % len(now_data))
                if ca >= 0 and ca % len(now_data) == most_common_idxs[cmn_idx][0] % len(now_data):
                    # 同じ枠に既に同じINDEXの候補が居る場合、TRUE
                    is_idx_existed = True
                    break

            if most_common_idxs[cmn_idx][0] in nearest_idxs or is_idx_existed:
                # 同じINDEXが既にリストにある場合
                # もしくは入れようとしているINDEXが反対枠の同じ並び順にいるか否か
                # logger.debug("次点繰り上げ cmn_idx:%s, val: %s, nearest_idxs: %s", cmn_idx, most_common_idxs[cmn_idx][0], nearest_idxs)
                # continue
                logger.debug("既出スキップ cmn_idx:%s, val: %s, nearest_idxs: %s", cmn_idx, most_common_idxs[cmn_idx][0], nearest_idxs)
                # 既出の場合、これ以上チェックできないので、次にいく
                cidx -= 1
                break
            elif most_common_idxs[cmn_idx][1] > 0:
                # 同じINDEXがリストにまだない場合
                logger.debug("採用 cmn_idx:%s, val: %s, nearest_idxs: %s", cmn_idx, most_common_idxs[cmn_idx][0], nearest_idxs)
                # 採用の場合、cidx減算
                is_passed = True
                cidx -= 1
                break
            else:
                logger.debug("再頻出ゼロ cmn_idx:%s, val: %s, nearest_idxs: %s", cmn_idx, most_common_idxs[cmn_idx][0], nearest_idxs)
                # 最頻出がない場合、これ以上チェックできないので、次にいく
                cidx -= 1
                break

        logger.debug("結果: near: %s, cmn_idx: %s, val: %s, most_common_idxs: %s", now_conf_idx, cmn_idx, most_common_idxs[cmn_idx][0], most_common_idxs)

        if is_passed:
            # 信頼度の高いINDEXに該当する最多ヒットINDEXを設定
            nearest_idxs[now_conf_idx] = most_common_idxs[cmn_idx][0]
        
        # 現在のループ回数は必ず加算
        cidxcnt += 1

        logger.debug("now_conf_idx: %s, cidx: %s, cidxcnt: %s, nearest_idxs: %s ---------------------", now_conf_idx, cidx, cidxcnt, nearest_idxs)

    logger.debug("nearest_idxs: %s", nearest_idxs)

    if -1 in nearest_idxs:
        # 不採用になって判定できなかったデータがある場合
        for _nidx, _nval in enumerate(nearest_idxs):
            if _nval == -1:
                # 該当値が-1(判定不可）の場合
                for _cidx in range(len(conf_idxs)):
                    logger.debug("_nidx: %s, _nval: %s, _cidx: %s, _cidx not in nearest_idxs: %s", _nidx, _nval, _cidx, _cidx not in nearest_idxs)
                    # INDEXを頭から順に見ていく（正0, 正1 ... 正n, 逆0, 逆1 ... 逆n)
                    if _cidx not in nearest_idxs:

                        # 入れようとしているINDEXが、採用枠（前半）か不採用枠（後半）か
                        if now_conf_idx < len(now_data):
                            # 採用枠(前半)の場合
                            check_ary = nearest_idxs[len(now_data): len(now_data)*2]
                        else:
                            # 不採用枠(後半)の場合
                            check_ary = nearest_idxs[0: len(now_data)]
                        
                        logger.debug("nearest_idxs: %s, _cidx: %s, check_ary: %s", nearest_idxs, _cidx, check_ary )

                        is_idx_existed = False
                        for ca in check_ary:
                            logger.debug("ca: %s, ca / len(now): %s, _cidx / len(now): %s", ca, ca % len(now_data), _cidx % len(now_data))
                            if ca >= 0 and ca % len(now_data) == _cidx % len(now_data):
                                # 同じ枠に既に同じINDEXの候補が居る場合、TRUE
                                is_idx_existed = True
                                break

                        if is_idx_existed == False:
                            # 該当INDEXがリストに無い場合、設定
                            nearest_idxs[_nidx] = _cidx
                            break

    logger.debug("is_upper_reverses: %s, is_lower_reverses: %s", is_upper_reverses, is_lower_reverses)
    logger.debug("past_sorted_idxs: %s nearest_idxs(retake): %s", past_sorted_idxs, nearest_idxs)

    # 最終的に人数分だけ残したINDEXリスト
    result_nearest_idxs = [-1 for x in range(len(now_data))]
    result_is_all_reverses = [False for x in range(len(now_data))]
    result_is_upper_reverses = [False for x in range(len(now_data))]
    result_is_lower_reverses = [False for x in range(len(now_data))]
    for _ridx in range(len(now_data)):
        # # 反転の可能性があるので、人数で割った余りを設定する
        sidx = int(nearest_idxs[_ridx] % len(now_data))

        if _ridx < len(now_data):
            # 自分より前に、自分と同じINDEXが居る場合、次のINDEXを引っ張り出す
            s = 1
            while sidx in result_nearest_idxs[0:_ridx+1]:
                newsidx = int(nearest_idxs[_ridx+s] % len(now_data))
                logger.debug("INDEX重複のため、次点繰り上げ: %s, sidx: %s, newsidx: %s", _ridx, sidx, newsidx)
                sidx = newsidx
                s += 1

        result_nearest_idxs[_ridx] = sidx
        result_is_upper_reverses[sidx] = is_upper_reverses[_ridx]
        result_is_lower_reverses[sidx] = is_lower_reverses[_ridx]
        result_is_all_reverses[sidx] = True if nearest_idxs[_ridx] >= len(now_data) and is_upper_reverses[_ridx] == False and is_lower_reverses[_ridx] == False else False

    logger.info("result_nearest_idxs: %s, all: %s, upper: %s, lower: %s", result_nearest_idxs, result_is_all_reverses, result_is_upper_reverses, result_is_lower_reverses)

    return result_nearest_idxs, result_is_all_reverses, result_is_upper_reverses, result_is_lower_reverses


# 過去データと現在データを比較して、頻出インデックス算出
def calc_most_common_idxs(conf_idxs, now_x, now_y, now_confs, past_x_ary, past_y_ary, past_conf_ary, idx_target=None):
    # 過去データの当該関節で、現在データと最も近いINDEXのリストを生成
    now_nearest_idxs = []
    most_common_idxs = []
    th = 0.3

    # X方向の頻出インデックス
    now_nearest_idxs, most_common_idxs, is_top = \
        calc_one_dimensional_most_common_idxs("x", conf_idxs, now_x, now_confs, past_x_ary, past_conf_ary, now_nearest_idxs, idx_target)

    if len(now_nearest_idxs) < len(now_x) * 0.2 or is_top == False:
        # チェック対象件数が2割未満、再頻出が70%未満、TOP2が別枠のいずれかの場合、
        # 位置データYを追加して、再度頻出チェック

        now_nearest_idxs, most_common_idxs, is_top = \
            calc_one_dimensional_most_common_idxs("y", conf_idxs, now_y, now_confs, past_y_ary, past_conf_ary, now_nearest_idxs, idx_target)

        return now_nearest_idxs, most_common_idxs, True, is_top

    return now_nearest_idxs, most_common_idxs, False, is_top

# 一方向だけの頻出インデックス算出
def calc_one_dimensional_most_common_idxs(dimensional, conf_idxs, now_datas, now_confs, past_datas, past_confs, now_nearest_idxs, idx_target=None):
    # 過去データの当該関節で、現在データと最も近いINDEXのリストを生成
    most_common_idxs = []
    th = 0.1

    # # 位置データ(全身＋手足)
    # for _idx in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,2,3,4,5,6,7,8,9,10,11,12,13]:
    # 位置データ(全身)
    for _idx in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]:
        one_data = now_datas[_idx]
        past_person = []
        for p, c in zip(past_datas, past_confs):
            # logger.debug("p: %s, c: %s", p, c)
            if _idx < len(p):
                pdata = 0
                if idx_target is None:
                    # logger.debug("c[_idx]: %s", c[_idx])
                    # pdata = 0 if c[_idx] < th else p[_idx]
                    pdata = p[_idx]
                else:
                    # logger.debug("c[idx_target[_idx]]: %s", c[idx_target[_idx]])
                    # pdata = 0 if c[idx_target[_idx]] < th else p[idx_target[_idx]]
                    pdata = p[idx_target[_idx]]
                
                past_person.append(pdata) 

        # 今回データがないものはチェック対象外
        if len(past_person) > 0 and 0 not in past_person and one_data > 0 and now_confs[_idx] > th:
            logger.debug("%s: %s, one_data %s", dimensional, past_person, one_data)
            now_nearest_idxs.append(get_nearest_idx(past_person, one_data))
        else:
            # logger.debug("%s:: past_person対象外: %s, x_data %s", dimensional, past_person, x_data)
            pass

    if len(now_nearest_idxs) > 0:
        most_common_idxs = Counter(now_nearest_idxs).most_common()

    # 頻出で振り分けた後、件数が足りない場合（全部どれか1つに寄せられている場合)
    if len(most_common_idxs) < len(conf_idxs):
        # logger.debug("頻出カウント不足: len(most_common_idxs): %s, len(conf_idxs): %s ", len(most_common_idxs), len(conf_idxs))
        for c in range(len(conf_idxs)):
            is_existed = False
            for m, mci in enumerate(most_common_idxs):
                if c == most_common_idxs[m][0]:
                    is_existed = True
                    break
            
            if is_existed == False:
                # 存在しないインデックスだった場合、追加                 
                most_common_idxs.append( (c, 0) )
    
    logger.debug("%s:: len(most_common_idxs): %s, len(conf_idxs): %s, len(now_nearest_idxs): %s, dimensional,len(now_datas): %s", dimensional, len(most_common_idxs), len(conf_idxs), len(now_nearest_idxs), len(now_datas))
    logger.debug("%s:: now_nearest_idxs: %s, most_common_idxs: %s", dimensional, now_nearest_idxs, most_common_idxs)

    sum_most_common_idxs, most_common_per, sum_top2_per, top_frame, second_frame = \
        get_most_common_frames(most_common_idxs, conf_idxs)

    # TOP1だけで7割か、TOP2で8割か
    is_top = most_common_idxs[0][1] > len(now_nearest_idxs) * 0.7 or (sum_top2_per > 0.8 and top_frame == second_frame)
    logger.debug("%s:: sum_top2_per: %s, len(now_datas): %s, top: %s, second: %s, is_top: %s", dimensional, sum_top2_per, int(len(conf_idxs)/2), top_frame, second_frame, is_top)

    return now_nearest_idxs, most_common_idxs, is_top

def get_most_common_frames(most_common_idxs, conf_idxs):

    sum_most_common_idxs = 0
    for mci_data in most_common_idxs:
        sum_most_common_idxs += mci_data[1]
    sum_top2_per = 0 if sum_most_common_idxs == 0 else (most_common_idxs[0][1] + most_common_idxs[1][1]) / sum_most_common_idxs
    top_frame = most_common_idxs[0][0] % int(len(conf_idxs)/2)

    # 下位が同率の場合、同じ枠があるかチェックする
    # １人の場合にも１回だけループを回すため、1から開始
    smidx = 1
    while smidx < len(most_common_idxs):
        if most_common_idxs[1][1] == most_common_idxs[smidx][1]:
            # ２位と３位以下が同率の場合
            second_frame = most_common_idxs[1][0] % int(len(conf_idxs)/2)
            third_frame = most_common_idxs[smidx][0] % int(len(conf_idxs)/2)
            # 1位と同じ枠を採用
            second_frame = third_frame if top_frame == third_frame else second_frame
            
            logger.debug("smidx: %s, top_frame: %s, second_frame: %s, third_frame: %s, most_common_idxs: %s", smidx, top_frame, second_frame, third_frame, most_common_idxs)

            smidx += 1
        else:
            second_frame = most_common_idxs[1][0] % int(len(conf_idxs)/2)
            break

    most_common_per = 0 if sum_most_common_idxs == 0 else most_common_idxs[0][1] / sum_most_common_idxs

    logger.debug("top_frame: %s, second_frame: %s, sum_most_common_idxs: %s, most_common_per: %s", top_frame, second_frame, sum_most_common_idxs, most_common_per)

    return sum_most_common_idxs, most_common_per, sum_top2_per, top_frame, second_frame


# 深度データで人物判定
def calc_depth_most_common_idxs(conf_idxs, now_depth, now_conf, past_depth_ary, past_conf_ary, now_nearest_idxs):
    # XYの頻出は引き継がない
    now_nearest_idxs = []
    most_common_idxs = []
    th = 0.1

    # 深度データY(全身＋体幹)
    for d_idx in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,0,1,2,6,8,11]:
        if d_idx < len(now_depth):
            d_data = now_depth[d_idx]
            past_depths = []
            for p in past_depth_ary:
                if d_idx < len(p):
                    # logger.debug("d_idx: %s, p[d_idx]: %s, c[d_idx]: %s", d_idx, p[d_idx], c[d_idx])
                    past_depths.append(p[d_idx]) 

            # 今回データがないものはチェック対象外
            if len(past_depths) > 0 and 0 not in past_depths and d_data > 0:
                logger.debug("past_depths: %s, d_data %s", past_depths, d_data)
                now_nearest_idxs.append(get_nearest_idx(past_depths, d_data))
            else:
                logger.debug("past_depths対象外: %s, d_data %s", past_depths, d_data)

    if len(now_nearest_idxs) > 0:
        most_common_idxs = Counter(now_nearest_idxs).most_common()

    logger.debug("d:: now_nearest_idxs: %s, most_common_idxs: %s, ", now_nearest_idxs, most_common_idxs)

    # logger.debug("past_depth_ary: %s", past_depth_ary)

    # past_depths = []
    # for p in past_depth_ary:
    #     past_sum_depths = []
    #     logger.debug("now_depth: %s", now_depth)
    #     for d_idx in range(len(now_depth)):
    #         logger.debug("d_idx: %s", d_idx)
    #         past_sum_depths.append(p[d_idx]) 

    #     logger.debug("past_sum_depths: %s", past_sum_depths)

    #     # 重み付けした平均値を求める
    #     past_depths.append(np.average(np.array(past_sum_depths), weights=[0.1,0.8,0.5,0.3,0.1,0.5,0.3,0.1,0.8,0.3,0.1,0.8,0.3,0.1,0.1,0.1,0.1,0.1]))
        
    #     # 今回データがないものはチェック対象外
    #     # if len(past_depths) > 0 and 0 not in past_depths and d_data > 0 and now_conf[d_idx] > th:
    #     #     logger.debug("[limbs] past_depths: %s, d_data %s", past_depths, d_data)
    #     #     now_nearest_idxs.append(get_nearest_idx(past_depths, d_data))

    # if len(now_nearest_idxs) > 0:
    #     most_common_idxs = Counter(now_nearest_idxs).most_common()

    # logger.debug("d:: now_nearest_idxs: %s, most_common_idxs: %s", now_nearest_idxs, most_common_idxs)

    # 頻出で振り分けた後、件数が足りない場合（全部どれか1つに寄せられている場合)
    if len(most_common_idxs) < len(conf_idxs):
        # logger.debug("頻出カウント不足: len(most_common_idxs): %s, len(conf_idxs): %s ", len(most_common_idxs), len(conf_idxs))
        for c in range(len(conf_idxs)):
            is_existed = False
            for m, mci in enumerate(most_common_idxs):
                if c == most_common_idxs[m][0]:
                    is_existed = True
                    break
            
            if is_existed == False:
                # 存在しないインデックスだった場合、追加                 
                most_common_idxs.append( (c, 0) )
    
    return now_nearest_idxs, most_common_idxs


def get_nearest_idx(target_list, num):
    """
    概要: リストからある値に最も近い値のINDEXを返却する関数
    @param target_list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値のINDEX
    """

    # logger.debug(target_list)
    # logger.debug(num)

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(target_list) - num).argmin()
    return idx


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
        # net.load(model_data_path, sess) 

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})
        
        # Plot result
        fig = plt.figure()
        ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        fig.colorbar(ii)
        plt.show()
        
        return pred
        
# Openposeから必要な部分だけ抽出
# 人体の位置が分かる事が重要なので、線形補間等は行わない
# Openposeの生のデータを使用する。
def read_openpose_json(openpose_output_dir):
    # openpose output format:
    # [x1,y1,c1,x2,y2,c2,...]
    # ignore confidence score, take x and y [x1,y1,x2,y2,...]

    openpose_filenames = []

    logger.info("start reading data: %s", openpose_output_dir)
    # load json files
    json_files = os.listdir(openpose_output_dir)
    # check for other file types
    json_files = sorted([filename for filename in json_files if filename.endswith(".json")])

    # jsonのファイル数が読み取り対象フレーム数
    _json_size = len(json_files)

    start_frame_index = 0
    is_started = False
    first_people_size = 1

    for file_name in json_files:
        logger.debug("reading {0}".format(file_name))
        _file = os.path.join(openpose_output_dir, file_name)
        if not os.path.isfile(_file): raise Exception("No file found!!, {0}".format(_file))
        try:
            data = json.load(open(_file))
        except Exception as e:
            logger.warn("JSON読み込み失敗のため、空データ読み込み, %s %s", _file, e)
            data = json.load(open("tensorflow/json/all_empty_keypoints.json"))

        # 12桁の数字文字列から、フレームINDEX取得
        frame_idx = int(re.findall("(\d{12})", file_name)[0])
        
        # ファイル名保持
        openpose_filenames.append(file_name)

        # 人数を計算
        _len = len(data["people"])

        if _len <= 0:
            # 人数が足りない場合、とりあえず空のを読み込んどく
            data = json.load(open("tensorflow/json/empty_keypoints.json"))

        # 人数分データ
        _multi_data = []

        if frame_idx <= 0 or is_started == False:
            # 最初のフレームはそのまま登録するため、INDEXをそのまま指定
            for pidx in range(_len):
                _multi_data.append(data["people"][pidx]["pose_keypoints_2d"])

            # 開始したらフラグを立てる
            is_started = True
            # 開始フレームインデックス保持
            start_frame_index = frame_idx
            # 人数保持
            first_people_size = _len

            # 初期化
            cache = [[[] for y in range(_len) ] for x in range(_json_size)]
            cache_confidence = [[[] for y in range(_len) ] for x in range(_json_size)]
        else:
            # 人数分セット
            for pidx in range(_len):
                _multi_data.append(data["people"][pidx]["pose_keypoints_2d"])
            
            epidx = 0
            while epidx + _len < first_people_size:
                # 足りない分、空データを加算する
                _multi_data.append(json.load(open("tensorflow/json/one_keypoints.json"))["pose_keypoints_2d"])

                epidx += 1

        # 配列用のインデックス
        _idx = frame_idx - start_frame_index

        # 人数分のデータを保持
        for _pidx, _data in enumerate(_multi_data):
            xy = []
            confidence = []
            for o in range(0,len(_data),3):
                xy.append(_data[o])
                xy.append(_data[o+1])
                confidence.append(_data[o+2])
        
            # logger.debug("found {0} for frame {1}".format(xy, str(frame_idx)))
            # add xy to frame
            cache[_idx][_pidx] = xy
            cache_confidence[_idx][_pidx] = confidence

    # logger.debug("cache: %s", cache)
    # logger.debug("cache_confidence: %s", cache_confidence)

    openpose_2d = [[[] for y in range(len(cache[0])) ] for x in range(_json_size)]
    logger.debug("len(openpose_2d): %s, len(openpose_2d[0]): %s", len(openpose_2d), len(openpose_2d[0]))

    # 推定用の関節位置を保持
    for frame, one_xy in enumerate(cache):
        for pidx, xy in enumerate(one_xy):
            # logger.debug("pidx: %s, xy: %s", pidx, len(xy))
            if len(xy) <= 0:
                # 1フレーム分のデータに関節情報がまったくない場合
                # 仮で全データにあり得ない値を入れておく
                openpose_2d[frame][pidx] = [ [float(-1), float(-1)] for x in range(int( len(cache[0][0]) /2)) ]
            else:
                # joints x,y array
                _len = len(xy) # 36

                # 1フレーム1人分のデータ
                one_xy_2d = []

                for x in range(0,_len,2):
                    # set x and y
                    y = x+1
                    
                    # logger.debug("pidx: %s, xy: %s", pidx, xy)
                    # logger.debug("conf: %s", cache_confidence[frame][pidx][int(x / 2)])
                    if cache_confidence[frame][pidx][int(x / 2)] > 0.3:
                        # 信頼度が一定以上の場合、関節の位置保持
                        one_xy_2d.append([float(xy[x]), float(xy[y])])
                    else:
                        # 信頼度が満たない場合、あり得ない値保持
                        one_xy_2d.append([float(-1), float(-1)])

                logger.debug("frame: %s, pidx: %s, xy: %s: %s", frame, pidx, x, one_xy_2d)
                openpose_2d[frame][pidx] = one_xy_2d

    return start_frame_index, openpose_2d, openpose_filenames
       
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
    parser.add_argument('--json_path', dest='json_path', help='openpose json result path', type=str)
    parser.add_argument('--now', dest='now', help='now', default=None, type=str)
    parser.add_argument('--interval', dest='interval', help='interval', type=int)
    parser.add_argument('--reverse_frames', dest='reverse_frames', help='reverse_frames', default="", type=str)
    parser.add_argument('--order_specific', dest='order_specific', help='order_specific', default="", type=str)
    parser.add_argument('--avi_output', dest='avi_output', help='avi_output', default='yes', type=str)
    parser.add_argument('--verbose', dest='verbose', help='verbose', type=int)
    args = parser.parse_args()

    logger.setLevel(level[args.verbose])

    # 間隔は1以上の整数
    interval = args.interval if args.interval > 0 else 1

    # AVI出力有無
    is_avi_output = False if args.avi_output == 'no' else True

    # 出力用日付
    if args.now is None:
        now_str = "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    else:
        now_str = args.now

    # 反転判定用辞書作成
    reverse_frame_dict = {}
    if args.reverse_frames is not None and len(args.reverse_frames) > 0:
        for frame in args.reverse_frames.split(','):
            # カンマで区切られている場合、各フレームを指定する
            if '-' in frame:
                # ハイフンで区切られている場合、フレーム範囲を指定する
                frange = frame.split('-')
                logger.debug("frange: %s", frange)
                if len(frange) >= 2 and frange[0].isdecimal() and frange[1].isdecimal():
                    for f in range(int(frange[0]), int(frange[1])+1):
                        # 指定フレームの辞書作成
                        reverse_frame_dict[f] = True
                else:
                    logger.warn("反転フレーム範囲指定失敗: [%s]", frame)
            elif frame.isdecimal():
                # 指定フレームの辞書作成
                reverse_frame_dict[int(frame)] = True
            else:
                logger.warn("反転フレーム範囲指定失敗: [%s]", frame)

        logger.info("反転フレームリスト: %s", reverse_frame_dict.keys())

    # 強制順番指定用辞書作成
    order_specific_dict = {}
    if args.order_specific is not None and len(args.order_specific) > 0:
        for frame in args.order_specific.split(']'):
            # 終わりカッコで区切る
            if ':' in frame:
                # コロンでフレーム番号と人物を区切る
                frames = frame.lstrip("[").split(':')[0]
                logger.info("frames: %s", frames)
                if '-' in frames:
                    frange = frames.split('-')
                    if len(frange) >= 2 and frange[0].isdecimal() and frange[1].isdecimal():
                        for f in range(int(frange[0]), int(frange[1])+1):
                            # 指定フレームの辞書作成
                            order_specific_dict[f] = []

                            for person_idx in frame.split(':')[1].split(','):
                                order_specific_dict[f].append(int(person_idx))
                else:        
                    if frames not in order_specific_dict:
                        # 該当フレームがまだない場合、作成
                        order_specific_dict[int(frames)] = []

                        for person_idx in frame.split(':')[1].split(','):
                            order_specific_dict[int(frames)].append(int(person_idx))
                

        logger.info("順番指定リスト: %s", order_specific_dict)

    # 日付+depthディレクトリ作成
    depth_path = '{0}/{1}_{2}_depth'.format(os.path.dirname(args.json_path), os.path.basename(args.json_path), now_str)
    
    os.makedirs(depth_path)

    # 関節二次元データを取得
    start_frame, openpose_2d, openpose_filenames = read_openpose_json(args.json_path)
    logger.info("開始フレームインデックス: %d", start_frame)


    op_avi_path = re.sub(r'_json$', "_openpose.avi", args.json_path)
    logger.debug("op_avi_path: %s", op_avi_path)
    # Openopse結果AVIを読み込む
    cnt = 0
    cap = cv2.VideoCapture(op_avi_path)

    # Predict the image
    predict_video(now_str, args.model_path, args.video_path, depth_path, interval, args.json_path, openpose_2d, openpose_filenames, start_frame, reverse_frame_dict, order_specific_dict, is_avi_output, args.verbose)

    logger.info("Done!!")
    logger.info("深度推定結果: {0}".format(depth_path +'/depth.txt'))

if __name__ == '__main__':
    main()

        



