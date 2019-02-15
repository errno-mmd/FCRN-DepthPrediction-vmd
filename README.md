# VMD-3d-pose-baseline-multi

このプログラムは、[FCRN-DepthPrediction](https://github.com/iro-cp/FCRN-DepthPrediction) \(Iro Laina様他\) を miu(miu200521358) がfork して、改造しました。

動作詳細等は上記URL、または [README-original.md](README-original.md) をご確認ください。

## 機能概要

- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) で検出された人体の骨格構造から、深度を推定します。
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)の関節XY位置情報と、深度推定を元に、複数人数のトレースで人物追跡を行います。

## 準備

詳細は、[Qiita](https://qiita.com/miu200521358/items/d826e9d70853728abc51)を参照して下さい。

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

1. [Openpose簡易起動バッチ](https://github.com/miu200521358/openpose-simple) で データを解析する
1. [OpenposeTo3D.bat](OpenposeTo3D.bat) を実行する
	- [OpenposeTo3D_en.bat](OpenposeTo3D_en.bat) is in English. !! The logs remain in Japanese.
1. [VideoToDepth.bat](VideoToDepth.bat) を実行する
	- [VideoToDepth_en.bat](VideoToDepth_en.bat) is in English. !! The logs remain in Japanese.
1. `解析対象映像ファイルパス` が聞かれるので、動画のファイルフルパスを入力する
1. `解析結果JSONディレクトリパス` が聞かれるので、1.の結果ディレクトリパスを指定する 
	- `{動画パス}/{動画ファイル名}_{実行年月日}/{動画ファイル名}_json` が対象ディレクトリパス
1. `深度推定間隔` が聞かれるので、深度推定を行うフレームの間隔(整数のみ)を指定する
    - 指定された間隔ごとに深度推定を行う
    - 未指定の場合、デフォルトで「10」とする
    - 値が小さいほど細かく深度推定を行うが、その分処理が遅くなる
1. `反転フレームリスト`が聞かれるので、Openposeが裏表を誤認識しているフレーム範囲を指定する。
	- ここで指定されたフレーム範囲内のみ、反転判定を行う。
	- `10,20` のように、カンマで区切って複数フレーム指定可能。
	- `10-15` のように、ハイフンで区切った場合、その範囲内のフレームが指定可能。
1. `順番指定リスト` が聞かれるので、交差後に人物追跡が間違っている場合に、フレームNoと人物インデックスの順番を指定する。
	- 人物インデックスは、0F目の左から0番目、1番目、と数える。
	- `[12:1,0]` と指定された場合、12F目は、画面左から、0F目の1番目、0F目の0番目と並び替える、とする。
	- `[12-15:1,0]` と指定された場合、12～15F目の範囲で、1番目・0番目と並び替える。
1. `詳細なログを出すか` 聞かれるので、出す場合、`yes` を入力する
    - 未指定 もしくは `no` の場合、通常ログ
1. 処理開始
1. 処理が終了すると、`解析結果JSONディレクトリパス`と同階層に以下結果が出力される
	- `{動画ファイル名}_json_{実行日時}_depth`
	    - depth.txt …　各関節位置の深度推定値リスト
	    - message.log …　出力順番等、パラメーター指定情報の出力ログ
	    - movie_depth.gif　…　深度推定の合成アニメーションGIF
	        - 白い点が関節位置として取得したポイントになる
	    - depth/depth_0000000000xx.png … 各フレームの深度推定結果
	    - ※複数人数のトレースを行った場合、全員分の深度情報が出力される
	- `{動画ファイル名}_json_{実行日時}_index{0F目の左からの順番}`
	    - depth.txt …　該当人物の各関節位置の深度推定値リスト
1. message.log に出力される情報
	- ＊＊05254F目の出力順番: [5254:1,0], 位置: {0: [552.915, 259.182], 1: [654.837, 268.902]}
		- 5254F目では、1, 0の順番に割り当てられた
			- 0番目に設定されている1は、[654.837, 268.902]の人物が推定された
			- 1番目に設定されている0は、[552.915, 259.182]の人物が推定された
		- このフレームの人物立ち位置が間違っている場合、[5254:0,1]を、`順番指定リスト`に指定すると、5254Fの出力順番が反転される
	- ※※03329F目 順番指定あり [1, 0]
		- 3229F目を、`順番指定リスト`で、[1,0]と指定してあり、それに準じて出力された
	- ※※04220F目 1番目の人物、下半身反転 [4220:1]
		- 4220F目を、`反転フレームリスト`で指定してあり、かつ反転判定された場合に反転出力された


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
【深度推定】：Iro Laina, miu200521358　…　https://github.com/miu200521358/FCRN-DepthPrediction-vmd
【Openpose→3D変換】：una-dinosauria, ArashHosseini, miu200521358　…　https://github.com/miu200521358/3d-pose-baseline-vmd
【Openpose→3D変換その2】：Dwango Media Village, miu200521358：MIT　…　https://github.com/miu200521358/3dpose_gan_vmd
【3D→VMD変換】： errno-mmd, miu200521358 　…　https://github.com/miu200521358/VMD-3d-pose-baseline-multi
```

- ニコニコ動画の場合、コンテンツツリーへ [トレース自動化マイリスト](https://www.nicovideo.jp/mylist/61943776) の最新版動画を登録してください。
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
