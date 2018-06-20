# VMD-3d-pose-baseline-multi

このプログラムは、[FCRN-DepthPrediction](https://github.com/iro-cp/FCRN-DepthPrediction) \(Iro Laina様他\) を miu(miu200521358) がfork して、改造しました。

動作詳細等は上記URL、または [README-original.md](README-original.md) をご確認ください。

## 機能概要

- 映像データと[miu200521358/3d-pose-baseline-vmd](https://github.com/miu200521358/3d-pose-baseline-vmd) で生成された2D関節データから、深度推定結果ファイルを生成します
- この機能はオプショナルです。深度推定結果ファイルがなかった場合、[miu200521358/VMD-3d-pose-baseline-multi](https://github.com/miu200521358/VMD-3d-pose-baseline-multi) はセンターZ判定を行いません。

## 準備

詳細は、[Qiita](https://qiita.com/miu200521358/items/d826e9d70853728abc51)を参照して下さい。(現在作成中)

### 依存関係

python3系 で以下をインストールして下さい

- [Tensorflow](https://www.tensorflow.org/)
- [OpenCV](http://opencv.org/)
- [tensorflow](https://www.tensorflow.org/) 1.0 or later

補足）以下プログラムが動作する環境であれば、追加インストール不要です。
 - [miu200521358/3d-pose-baseline-vmd](https://github.com/miu200521358/3d-pose-baseline-vmd)
 - [miu200521358/VMD-3d-pose-baseline-multi](https://github.com/miu200521358/VMD-3d-pose-baseline-multi)

## モデルデータ

[tensorflow用モデルデータ](http://campar.in.tum.de/files/rupprecht/depthpred/NYU_FCRN-checkpoint.zip)を「`tensorflow/data`」ディレクトリを作成し、以下に配置する

## 実行方法

1. [miu200521358/3d-pose-baseline-vmd](https://github.com/miu200521358/3d-pose-baseline-vmd) で生成された2D関節データ (smoothed.txt) を用意する
1. [VideoToDepth.bat](VideoToDepth.bat) を実行する
1. `解析対象映像ファイルパス` が聞かれるので、動画のファイルフルパスを入力する
1. `3D解析結果ディレクトリパス` が聞かれるので、1.の結果ディレクトリパスを指定する
1. `深度推定間隔` が聞かれるので、深度推定を行うフレームの間隔(整数のみ)を指定する
    - 指定された間隔ごとに深度推定を行う
    - 未指定の場合、デフォルトで「10」とする
    - 値が小さいほど細かく深度推定を行うが、その分処理が遅くなる
1. `詳細なログを出すか` 聞かれるので、出す場合、`yes` を入力する
    - 未指定 もしくは `no` の場合、通常ログ
1. 処理開始
1. 処理が終了すると、1. の結果ディレクトリ以下に結果が出力される
    - depth.txt …　腰位置の深度推定値リスト
    - movie_depth.gif　…　深度推定の合成アニメーションGIF
        - 白い点が腰位置として取得したポイントになる
    - depth/depth_0000000000xx.png … 各フレームの深度推定結果

## ライセンス
Simplified BSD License

以下の行為は自由に行って下さい

- モーションの調整・改変
- ニコニコ動画やTwitter等へのモーション使用動画投稿
- モーションの不特定多数への配布
    - **必ず踊り手様や各権利者様に失礼のない形に調整してください**

以下の行為は必ず行って下さい。ご協力よろしくお願いいたします。

- クレジットへの記載を、テキストで検索できる形で記載お願いします。

```
ツール名：モーショントレース自動化キット
権利者名：miu200521358
```

- モーションを配布する場合、以下ライセンスを同梱してください。 (記載場所不問)

```
LICENCE

モーショントレース自動化キット
【Openpose】：CMU　…　https://github.com/CMU-Perceptual-Computing-Lab/openpose
【Openpose起動バッチ】：miu200521358　…　https://github.com/miu200521358/openpose-simple
【Openpose→3D変換】：una-dinosauria, ArashHosseini, miu200521358　…　https://github.com/miu200521358/3d-pose-baseline-vmd
【深度推定】：Iro Laina, miu200521358　…　https://github.com/miu200521358/FCRN-DepthPrediction-vmd
【3D→VMD変換】： errno-mmd, miu200521358 　…　https://github.com/miu200521358/VMD-3d-pose-baseline-multi
```

- ニコニコ動画の場合、コンテンツツリーへ動画\([sm33272961](http://www.nicovideo.jp/watch/sm33272961)\)を登録してください。
    - コンテンツツリーに登録していただける場合、テキストでのクレジット有無は問いません。

以下の行為はご遠慮願います

- 自作発言
- 権利者様のご迷惑になるような行為
- 営利目的の利用
- 他者の誹謗中傷目的の利用（二次元・三次元不問）
- 過度な暴力・猥褻・恋愛・猟奇的・政治的・宗教的表現を含む（R-15相当）作品への利用
- その他、公序良俗に反する作品への利用

## 免責事項

- 自己責任でご利用ください
- ツール使用によって生じたいかなる問題に関して、作者は一切の責任を負いかねます
