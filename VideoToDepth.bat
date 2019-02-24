@echo off
rem --- 
rem ---  映像データから深度推定を行う
rem --- 

rem ---  カレントディレクトリを実行先に変更
cd /d %~dp0

rem ---  入力対象映像ファイルパス
echo Openposeで解析した映像のファイルのフルパスを入力して下さい。
echo この設定は半角英数字のみ設定可能で、必須項目です。
set INPUT_VIDEO=
set /P INPUT_VIDEO=■解析対象映像ファイルパス: 
rem echo INPUT_VIDEO：%INPUT_VIDEO%

IF /I "%INPUT_VIDEO%" EQU "" (
    ECHO 解析対象映像ファイルパスが設定されていないため、処理を中断します。
    EXIT /B
)

rem ---  解析結果JSONディレクトリパス
echo Openposeの解析結果のJSONディレクトリのフルパスを入力して下さい。({動画名}_json)
echo この設定は半角英数字のみ設定可能で、必須項目です。
set OPENPOSE_JSON=
set /P OPENPOSE_JSON=■解析結果JSONディレクトリパス: 
rem echo OPENPOSE_JSON：%OPENPOSE_JSON%

IF /I "%OPENPOSE_JSON%" EQU "" (
    ECHO 解析結果JSONディレクトリパスが設定されていないため、処理を中断します。
    EXIT /B
)

rem ---  深度推定間隔
echo --------------
set DEPTH_INTERVAL=10
echo 深度推定を行うフレームの間隔を数値で入力して下さい。
echo 値が小さいほど、細かく深度推定を行います。（その分、時間がかかります）
echo 何も入力せず、ENTERを押下した場合、「%DEPTH_INTERVAL%」間隔で処理します。
set /P DEPTH_INTERVAL="■深度推定間隔: "

rem ---  反転フレームリスト
echo --------------
set REVERSE_FRAME_LIST=
echo Openposeが誤認識して反転しているフレーム番号(0始まり)を指定してください。
echo ここで指定された番号のフレームに対して、反転判定を行い、反転認定された場合、関節位置が反転されます。
echo カンマで複数件指定可能です。また、ハイフンで範囲が指定可能です。
echo 例）4,10-12　…　4,10,11,12 が反転判定対象フレームとなります。
set /P REVERSE_FRAME_LIST="■反転フレームリスト: "

rem ---  順番指定リスト
echo --------------
set ORDER_SPECIFIC_LIST=
echo 複数人数トレースで、交差後の人物INDEX順番を指定してください。
echo フォーマット：［＜フレーム番号＞:左から0番目にいる人物のインデックス,左から1番目…］
echo 人物インデックスは、Openposeの出力結果JSONの出力順に対応しています。
echo 例）[10:1,0]　…　10F目は、左から1番目の人物、0番目の人物の順番に並べ替えます。
echo [10:1,0][30:0,1]のように、カッコ単位で複数件指定可能です。
echo 例）[10-15:1,0][30:0,1]　…　10～15F目: 1, 0の順番、30F目: 0, 1の順番。
set /P ORDER_SPECIFIC_LIST="■順番指定リスト: "

rem ---  MMD用AVI出力
echo --------------
echo MMD用AVIを出すか、yes か no を入力して下さい。
echo MMD用AVIは、Openposeの結果に、人物INDEX別情報を乗せて、サイズ小さめで出力します。
echo コーデックは「IYUV」「I420」のいずれかです。
echo 何も入力せず、ENTERを押下した場合、MMD用AVIを出力します。
set AVI_OUTPUT=yes
set /P AVI_OUTPUT="■MMD用AVI[yes/no]: "

rem ---  詳細ログ有無
echo --------------
echo 詳細なログを出すか、yes か no を入力して下さい。
echo 何も入力せず、ENTERを押下した場合、通常ログと深度推定GIFを出力します。
echo warn と指定すると、アニメーションGIFも出力しません。（その分早いです）
set VERBOSE=2
set IS_DEBUG=no
set /P IS_DEBUG="■詳細ログ[yes/no/warn]: "

IF /I "%IS_DEBUG%" EQU "yes" (
    set VERBOSE=3
)

IF /I "%IS_DEBUG%" EQU "warn" (
    set VERBOSE=1
)

rem ---  python 実行
python tensorflow/predict_video.py --model_path tensorflow/data/NYU_FCRN.ckpt --video_path %INPUT_VIDEO% --json_path %OPENPOSE_JSON% --interval %DEPTH_INTERVAL% --reverse_frames "%REVERSE_FRAME_LIST%" --order_specific "%ORDER_SPECIFIC_LIST%" --avi_output %AVI_OUTPUT% --verbose %VERBOSE%


