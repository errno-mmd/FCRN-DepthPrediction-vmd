@echo off
rem --- 
rem ---  Perform depth estimation from video data
rem --- 

rem ---  Change the current directory to the execution destination
cd /d %~dp0

rem ---  Input target video file path
echo Please enter the full path of the file of the video analyzed with Openpose.
echo This setting is available only for half size alphanumeric characters, it is a required item.
set INPUT_VIDEO=
set /P INPUT_VIDEO=** Movie file path to be analyzed: 
rem echo INPUT_VIDEOÅF%INPUT_VIDEO%

IF /I "%INPUT_VIDEO%" EQU "" (
    ECHO Processing is suspended because the analysis target video file path is not set.
    EXIT /B
)

rem ---  3d-pose-baseline-vmd Analysis result JSON directory path
echo Please enter the absolute path of the analysis result directory of 3d-pose-baseline-vmd.(3d_{yyyymmdd_hhmmss}_idx00)
echo This setting is available only for half size alphanumeric characters, it is a required item.
set TARGET_BASELINE_DIR=
set /P TARGET_BASELINE_DIR=** 3D analysis result directory path: 
rem echo TARGET_DIRÅF%TARGET_DIR%

IF /I "%TARGET_BASELINE_DIR%" EQU "" (
    ECHO 3D analysis result directory path is not set, processing will be interrupted.
    EXIT /B
)

rem ---  Depth estimation interval
echo --------------
set DEPTH_INTERVAL=10
echo Please enter the interval of the frame to be estimated depth.
echo The smaller the value, the finer the depth estimation. (It takes time to do so)
echo If nothing is entered and ENTER is pressed, processing is done at the interval of "%DEPTH_INTERVAL%".
set /P DEPTH_INTERVAL="** Depth estimation interval: "

rem ---  Presence of detailed log

echo --------------
echo Please output detailed logs or enter yes or no.
echo If nothing is entered and ENTER is pressed, normal log and depth estimation GIF are output.
echo If warn is specified, animation GIF is not output. (That is earlier)
set VERBOSE=2
set IS_DEBUG=no
set /P IS_DEBUG="** Detailed log[yes/no/warn]: "

IF /I "%IS_DEBUG%" EQU "yes" (
    set VERBOSE=3
)

IF /I "%IS_DEBUG%" EQU "warn" (
    set VERBOSE=1
)

rem ---  python run
python tensorflow/predict_video.py --model_path tensorflow/data/NYU_FCRN.ckpt --video_path %INPUT_VIDEO% --baseline_path %TARGET_BASELINE_DIR% --interval %DEPTH_INTERVAL% --verbose %VERBOSE%


