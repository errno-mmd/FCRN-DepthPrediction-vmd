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

rem ---  ��͌���JSON�f�B���N�g���p�X
echo Openpose�̉�͌��ʂ�JSON�f�B���N�g���̃t���p�X����͂��ĉ������B({���於}_json)
echo ���̐ݒ�͔��p�p�����̂ݐݒ�\�ŁA�K�{���ڂł��B
set OPENPOSE_JSON=
set /P OPENPOSE_JSON=����͌���JSON�f�B���N�g���p�X: 
rem echo OPENPOSE_JSON�F%OPENPOSE_JSON%

IF /I "%OPENPOSE_JSON%" EQU "" (
    ECHO ��͌���JSON�f�B���N�g���p�X���ݒ肳��Ă��Ȃ����߁A�����𒆒f���܂��B
    EXIT /B
)

rem ---  �[�x����Ԋu
echo --------------
set DEPTH_INTERVAL=10
echo �[�x������s���t���[���̊Ԋu�𐔒l�œ��͂��ĉ������B
echo �l���������قǁA�ׂ����[�x������s���܂��B�i���̕��A���Ԃ�������܂��j
echo �������͂����AENTER�����������ꍇ�A�u%DEPTH_INTERVAL%�v�Ԋu�ŏ������܂��B
set /P DEPTH_INTERVAL="���[�x����Ԋu: "

rem ---  ���]�t���[�����X�g
echo --------------
set REVERSE_FRAME_LIST=
echo Openpose����F�����Ĕ��]���Ă���t���[���ԍ�(0�n�܂�)���w�肵�Ă��������B
echo �����Ŏw�肳�ꂽ�ԍ��̃t���[���ɑ΂��āA���]������s���A���]�F�肳�ꂽ�ꍇ�A�֐߈ʒu�����]����܂��B
echo �J���}�ŕ������w��\�ł��B�܂��A�n�C�t���Ŕ͈͂��w��\�ł��B
echo ��j4,10-12�@�c�@4,10,11,12 �����]����Ώۃt���[���ƂȂ�܂��B
set /P REVERSE_FRAME_LIST="�����]�t���[�����X�g: "

rem ---  ���Ԏw�胊�X�g
echo --------------
set ORDER_SPECIFIC_LIST=
echo �����l���g���[�X�ŁA������̐l��INDEX���Ԃ��w�肵�Ă��������B
echo �t�H�[�}�b�g�F�m���t���[���ԍ���:������0�Ԗڂɂ���l���̃C���f�b�N�X,������1�Ԗځc�n
echo �l���C���f�b�N�X�́AOpenpose�̏o�͌���JSON�̏o�͏��ɑΉ����Ă��܂��B
echo ��j[10:1,0]�@�c�@10F�ڂ́A������1�Ԗڂ̐l���A0�Ԗڂ̐l���̏��Ԃɕ��בւ��܂��B
echo [10:1,0][30:0,1]�̂悤�ɁA�J�b�R�P�ʂŕ������w��\�ł��B
echo ��j[10-15:1,0][30:0,1]�@�c�@10�`15F��: 1, 0�̏��ԁA30F��: 0, 1�̏��ԁB
set /P ORDER_SPECIFIC_LIST="�����Ԏw�胊�X�g: "

rem ---  MMD�pAVI�o��
echo --------------
echo MMD�pAVI���o�����Ayes �� no ����͂��ĉ������B
echo MMD�pAVI�́AOpenpose�̌��ʂɁA�l��INDEX�ʏ����悹�āA�T�C�Y�����߂ŏo�͂��܂��B
echo �R�[�f�b�N�́uIYUV�v�uI420�v�̂����ꂩ�ł��B
echo �������͂����AENTER�����������ꍇ�AMMD�pAVI���o�͂��܂��B
set AVI_OUTPUT=yes
set /P AVI_OUTPUT="��MMD�pAVI[yes/no]: "

rem ---  �ڍ׃��O�L��
echo --------------
echo �ڍׂȃ��O���o�����Ayes �� no ����͂��ĉ������B
echo �������͂����AENTER�����������ꍇ�A�ʏ탍�O�Ɛ[�x����GIF���o�͂��܂��B
echo warn �Ǝw�肷��ƁA�A�j���[�V����GIF���o�͂��܂���B�i���̕������ł��j
set VERBOSE=2
set IS_DEBUG=no
set /P IS_DEBUG="���ڍ׃��O[yes/no/warn]: "

IF /I "%IS_DEBUG%" EQU "yes" (
    set VERBOSE=3
)

IF /I "%IS_DEBUG%" EQU "warn" (
    set VERBOSE=1
)

rem ---  python ���s
python tensorflow/predict_video.py --model_path tensorflow/data/NYU_FCRN.ckpt --video_path %INPUT_VIDEO% --json_path %OPENPOSE_JSON% --interval %DEPTH_INTERVAL% --reverse_frames "%REVERSE_FRAME_LIST%" --order_specific "%ORDER_SPECIFIC_LIST%" --avi_output %AVI_OUTPUT% --verbose %VERBOSE%


