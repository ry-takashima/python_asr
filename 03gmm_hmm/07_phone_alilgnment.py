# -*- coding: utf-8 -*-

#
# HMMモデルで音素アライメントを推定します．
# 推定したアライメントをスペクトログラム中に
# 図示します．
#

# hmmfunc.pyからMonoPhoneHMMクラスをインポート
from hmmfunc import MonoPhoneHMM

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# wavデータを読み込むためのモジュール(wave)をインポート
import wave

# プロット用モジュール(matplotlib)をインポート
import matplotlib.pyplot as plt

# os, sysモジュールをインポート
import sys
import os

#
# メイン関数
#
if __name__ == "__main__":

    # HMMファイル
    hmm_file = './exp/model_3state_1mix/10.hmm'

    # 評価データの特徴量リストのファイル
    feat_scp = './exp/data/test/mfcc/feats.scp'

    # wavデータリストのファイル
    wav_scp = 'exp/data/test/wav.scp'

    # 評価データのラベルファイル
    label_file = './exp/data/test/text_int'

    # 評価する発話ID
    eval_utt = '7'

    # フレームサイズ [ミリ秒]
    frame_size = 25
    # フレームシフト [ミリ秒]
    # 特徴量抽出時に指定した値と同じものを使用する
    frame_shift = 10

    # プロットを出力するファイル(pngファイル)
    out_plot = './alignment.png'

    #
    # 処理ここから
    #

    # MonoPhoneHMMクラスを呼び出す
    hmm = MonoPhoneHMM()

    # HMMを読み込む
    hmm.load_hmm(hmm_file)

    # ラベルファイルを開き，発話ID毎のラベル情報を得る
    label_list = {}
    with open(label_file, mode='r') as f:
        for line in f:
            # 0列目は発話ID
            utt = line.split()[0]
            # 1列目以降はラベル
            lab = line.split()[1:]
            # 各要素は文字として読み込まれているので，
            # 整数値に変換する
            lab = np.int64(lab)
            # label_listに登録
            label_list[utt] = lab

    # 特徴量リストファイルを開き，
    # 発話ID毎の特徴量ファイルのパスを得る
    feat_list = {}
    with open(feat_scp, mode='r') as f:
        for line in f:
            # 0列目は発話ID
            utt = line.split()[0]
            # 1列目はファイルパス
            ff = line.split()[1]
            # 3列目は次元数
            nd = int(line.split()[3])
            # 次元数がHMMの次元数と一致しなければエラー
            if hmm.num_dims != nd:
                sys.stderr.write(\
                    '%s: unexpected #dims (%d)\n'\
                    % (utt, nd))
                exit(1)
            # feat_listに登録
            feat_list[utt] = ff

    # wavリストファイルを開き，
    # 発話ID毎のwavファイルのパスを得る
    wav_list = {}
    with open(wav_scp, mode='r') as f:
        for line in f:
            # 0列目は発話ID
            utt = line.split()[0]
            # 1列目はファイルパス
            wf = line.split()[1]
            # wav_listに登録
            wav_list[utt] = wf

    # 評価する発話IDがリストになければエラー
    if (not eval_utt in label_list) \
      or (not eval_utt in feat_list) \
      or (not eval_utt in wav_list):
        sys.stderr.write('invalid eval_utt\n')
        exit(1)

    
    # ラベルを得る
    label = label_list[eval_utt]
    
    # 特徴量ファイル名を得る
    feat_file = feat_list[eval_utt]

    # 特徴量ファイルを開く
    feat = np.fromfile(feat_file, dtype=np.float32)
    # フレーム数 x 次元数の配列に変形
    feat = feat.reshape(-1, hmm.num_dims)

    # 音素アライメントを推定する
    # alignmentにはフレーム毎の音素ラベルが格納される
    alignment = hmm.phone_alignment(feat, label)
    
    #
    # 結果を出力する
    #
    # 音素境界のリスト
    boundary_list = []
    # 一つ前のフレームの音素
    prev_phone = ""
    # alignmentの音素を一つずつ読み込む
    for n, phone in enumerate(alignment):
        # 最初のフレームの場合
        if n == 0:
            boundary_time = 0
            # 音素と開始時刻を表示
            sys.stdout.write('%s %f ' \
                % (phone, boundary_time))
            # 前の音素を更新
            prev_phone = phone
            # 音素境界情報の追加
            boundary_list.append((phone, boundary_time))
            continue

        # 一つ前のフレームの音素と異なる場合
        if phone != prev_phone:
            # フレーム番号を秒に変換する
            boundary_time = n * frame_shift * 0.001
            # 前の音素の終了時刻を表示
            sys.stdout.write('%f\n' \
                % (boundary_time))
            # 音素と開始時刻を表示
            sys.stdout.write('%s %f ' \
                % (phone, boundary_time))
            # 前の音素を更新
            prev_phone = phone
            # 音素境界情報の追加
            boundary_list.append((phone, boundary_time))

        # 最終フレームの場合
        if n == len(alignment) - 1:
            # フレーム番号を秒に変換する
            boundary_time = n * frame_shift * 0.001
            # 前の音素の終了時刻を表示
            sys.stdout.write('%f\n' \
                % (boundary_time))

    #
    # アライメント結果を元に
    # スペクトログラムを図示する
    # 
    # wavファイルを得る
    wav_file = wav_list[eval_utt]

    # wavファイルを開き、以降の処理を行う
    with wave.open(wav_file) as wav:
        # wavデータを読み込む
        sample_frequency = wav.getframerate()
        num_samples = wav.getnframes()
        waveform = wav.readframes(num_samples)
        waveform = np.frombuffer(waveform, dtype=np.int16)

    # フレームサイズをミリ秒からサンプル数に変換
    frame_size = int(sample_frequency \
                     * frame_size * 0.001)
    # フレームシフトをミリ秒からサンプル数へ変換
    frame_shift = int(sample_frequency * frame_shift * 0.001)

    # FFTを行う範囲のサンプル数を，
    # フレームサイズ以上の2のべき乗に設定
    fft_size = 1
    while fft_size < frame_size:
        fft_size *= 2

    # 短時間フーリエ変換をしたときの
    # 総フレーム数を計算する
    num_frames = (num_samples - frame_size) \
               // frame_shift + 1

    # スペクトログラムを計算する
    spectrogram = np.zeros((num_frames, fft_size))
    for frame_idx in range(num_frames):
        # 1フレーム分の波形を抽出
        start_index = frame_idx * frame_shift
        frame = waveform[start_index : \
                         start_index + frame_size].copy()

        # 対数振幅スペクトルを計算
        frame = frame * np.hamming(frame_size)
        spectrum = np.fft.fft(frame, n=fft_size)
        log_absolute = np.log(np.abs(spectrum) + 1E-7)
        spectrogram[frame_idx, :] = log_absolute

    # プロットの描画領域を作成
    plt.figure(figsize=(10,10))

    # 描画領域を縦に2分割し、
    # 上側に時間波形をプロットする
    plt.subplot(2, 1, 1)
    time_axis = np.arange(num_samples) / sample_frequency
    plt.plot(time_axis, waveform, color='k')

    # waveformの最大値を元に縦軸の最大値を決める
    ymax = np.max(np.abs(waveform))*1.05

    # 音素境界を線で引く
    for (p, b) in boundary_list:
        # 音素境界の時刻で縦線を引く
        plt.vlines(b, -ymax, ymax, 
                   linestyle='dashed', color='k')
        # 音素ラベルを表示させる
        plt.text(b, ymax+50 , p, fontsize=14)

    plt.xlabel('Time [sec]')
    plt.ylabel('Value')
    plt.ylim([-ymax, ymax])
    plt.xlim([0, num_samples / sample_frequency])

    # 2分割された描画領域の下側に
    # スペクトログラムをプロットする
    plt.subplot(2, 1, 2)
    # カラーマップのレンジを調整
    spectrogram -= np.max(spectrogram)
    vmax = np.abs(np.min(spectrogram)) * 0.0
    vmin = - np.abs(np.min(spectrogram)) * 0.7
    plt.imshow(spectrogram.T[-1::-1,:], 
               extent=[0, num_samples / sample_frequency, 
                       0, sample_frequency],
               cmap = 'gray',
               vmax = vmax,
               vmin = vmin,
               aspect = 'auto')

    # 音素境界を線で引く
    for (p, b) in boundary_list:
        # 音素境界の時刻で縦線を引く
        plt.vlines(b, 0, sample_frequency, 
                   linestyle='dashed', color='w')
        # 音素ラベルを表示させる
        plt.text(b, sample_frequency/2+50 , p, fontsize=14)
    
    # 横軸と縦軸のラベルを定義
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.ylim([0, sample_frequency/2])

    # プロットを保存する
    plt.savefig(out_plot)
