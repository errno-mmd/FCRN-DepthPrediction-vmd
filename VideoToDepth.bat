@echo off
rem --- 
rem ---  �f���f�[�^����[�x������s��
rem --- 

rem ---  �J�����g�f�B���N�g�������s��ɕύX
cd /d %~dp0

rem ---  ���͑Ώۉf���t�@�C���p�X
echo Openpose�ŉ�͂����f���̃t�@�C���̃t���p�X����͂��ĉ������B
echo ���̐ݒ�͔��p�p�����̂ݐݒ�\�ŁA�K�{���ڂł��B
set INPUT_VIDEO=
set /P INPUT_VIDEO=����͑Ώۉf���t�@�C���p�X: 
rem echo INPUT_VIDEO�F%INPUT_VIDEO%

IF /I "%INPUT_VIDEO%" EQU "" (
    ECHO ��͑Ώۉf���t�@�C���p�X���ݒ肳��Ă��Ȃ����߁A�����𒆒f���܂��B
    EXIT /B
)

rem ---  3d-pose-baseline-vmd��͌���JSON�f�B���N�g���p�X
echo 3d-pose-baseline-vmd�̉�͌��ʃf�B���N�g���̐�΃p�X����͂��ĉ������B(3d_{���s����}_idx00)
echo ���̐ݒ�͔��p�p�����̂ݐݒ�\�ŁA�K�{���ڂł��B
set TARGET_BASELINE_DIR=
set /P TARGET_BASELINE_DIR=��3D��͌��ʃf�B���N�g���p�X: 
rem echo TARGET_DIR�F%TARGET_DIR%

IF /I "%TARGET_BASELINE_DIR%" EQU "" (
    ECHO 3D��͌��ʃf�B���N�g���p�X���ݒ肳��Ă��Ȃ����߁A�����𒆒f���܂��B
    EXIT /B
)

rem ---  �[�x����Ԋu
echo --------------
set DEPTH_INTERVAL=10
echo �[�x������s���t���[���̊Ԋu�𐔒l�œ��͂��ĉ������B
echo �l���������قǁA�ׂ����[�x������s���܂��B�i���̕��A���Ԃ�������܂��j
echo �������͂����AENTER�����������ꍇ�A�u%DEPTH_INTERVAL%�v�Ԋu�ŏ������܂��B
set /P DEPTH_INTERVAL="�[�x����Ԋu: "

rem ---  �ڍ׃��O�L��

echo --------------
echo �ڍׂȃ��O���o�����Ayes �� no ����͂��ĉ������B
echo �������͂����AENTER�����������ꍇ�A�ʏ탍�O�Ɛ[�x����GIF���o�͂��܂��B
set VERBOSE=2
set IS_DEBUG=no
set /P IS_DEBUG="�ڍ׃��O[yes/no]: "

IF /I "%IS_DEBUG%" EQU "yes" (
    set VERBOSE=3
)

rem ---  python ���s
python tensorflow/predict_video.py --model_path tensorflow/data/NYU_FCRN.ckpt --video_path %INPUT_VIDEO% --baseline_path %TARGET_BASELINE_DIR% --interval %DEPTH_INTERVAL% --verbose %VERBOSE%


