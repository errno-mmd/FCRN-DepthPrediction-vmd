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

rem ---  3d-pose-baseline-vmd解析結果JSONディレクトリパス
echo 3d-pose-baseline-vmdの解析結果ディレクトリの絶対パスを入力して下さい。(3d_{実行日時}_idx00)
echo この設定は半角英数字のみ設定可能で、必須項目です。
set TARGET_BASELINE_DIR=
set /P TARGET_BASELINE_DIR=■3D解析結果ディレクトリパス: 
rem echo TARGET_DIR：%TARGET_DIR%

IF /I "%TARGET_BASELINE_DIR%" EQU "" (
    ECHO 3D解析結果ディレクトリパスが設定されていないため、処理を中断します。
    EXIT /B
)

rem ---  深度推定間隔
echo --------------
set DEPTH_INTERVAL=10
echo 深度推定を行うフレームの間隔を数値で入力して下さい。
echo 値が小さいほど、細かく深度推定を行います。（その分、時間がかかります）
echo 何も入力せず、ENTERを押下した場合、「%DEPTH_INTERVAL%」間隔で処理します。
set /P DEPTH_INTERVAL="深度推定間隔: "

rem ---  詳細ログ有無

echo --------------
echo 詳細なログを出すか、yes か no を入力して下さい。
echo 何も入力せず、ENTERを押下した場合、通常ログと深度推定GIFを出力します。
set VERBOSE=2
set IS_DEBUG=no
set /P IS_DEBUG="詳細ログ[yes/no]: "

IF /I "%IS_DEBUG%" EQU "yes" (
    set VERBOSE=3
)

rem ---  python 実行
python tensorflow/predict_video.py --model_path tensorflow/data/NYU_FCRN.ckpt --video_path %INPUT_VIDEO% --baseline_path %TARGET_BASELINE_DIR% --interval %DEPTH_INTERVAL% --verbose %VERBOSE%


