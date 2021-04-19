# -*- coding: utf-8 -*-

#
# 短時間フーリエ変換を用いて
# 音声のスペクトログラムを作成します．
#

# wavデータを読み込むためのモジュール(wave)をインポート
import wave

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# プロット用モジュール(matplotlib)をインポート
import matplotlib.pyplot as plt

#
# メイン関数
#
if __name__ == "__main__":
    # 開くwavファイル
    wav_file_1 = './wav/REPEAT500_set1_009.wav'
    wav_file_2 = './wav/REPEAT500_set2_009.wav'

    # サンプリング周波数
    sample_frequency = 16000
    # フレームサイズ [ミリ秒]
    frame_size = 25
    # フレームシフト [ミリ秒]
    frame_shift = 10

    # プロットを出力するファイル(pngファイル)
    out_plot = './spectrogram.png'

    # フレームサイズをミリ秒からサンプル数に変換
    frame_size = int(sample_frequency * frame_size * 0.001)

    # フレームシフトをミリ秒からサンプル数へ変換
    frame_shift = int(sample_frequency * frame_shift * 0.001)

    # FFTを行う範囲のサンプル数を，
    # フレームサイズ以上の2のべき乗に設定
    fft_size = 1
    while fft_size < frame_size:
        fft_size *= 2

    #
    # 時間の縮尺を統一してスペクトログラムを
    # 表示させるため，時間を長い方に統一する
    # 
    max_num_samples = 0
    for n, wav_file in enumerate([wav_file_1, wav_file_2]):
        with wave.open(wav_file) as wav:
            num_samples = wav.getnframes()
        max_num_samples = max([num_samples, max_num_samples])
    # 最大時間に対するフレーム数を計算
    max_num_frames = (max_num_samples - frame_size) // frame_shift + 1

    # プロットの描画領域を作成
    plt.figure(figsize=(10,10))

    # 2個のwavfileに対して以下を実行
    for n, wav_file in enumerate([wav_file_1, wav_file_2]):
        # wavファイルを開き、以降の処理を行う
        with wave.open(wav_file) as wav:
            # wavデータの情報を読み込む
            num_samples = wav.getnframes()
            waveform = wav.readframes(num_samples)
            waveform = np.frombuffer(waveform, dtype=np.int16)

        # 短時間フーリエ変換をしたときの
        # 総フレーム数を計算する
        num_frames = (num_samples - frame_size) // frame_shift + 1

        # スペクトログラムの行列を用意
        # フレーム数は最大フレーム数にする
        spectrogram = np.zeros((max_num_frames, fft_size))

        # 1フレームずつ振幅スペクトルを計算する
        for frame_idx in range(num_frames):
            # 分析の開始位置は，フレーム番号(0始まり)*フレームシフト
            start_index = frame_idx * frame_shift

            # 1フレーム分の波形を抽出
            frame = waveform[start_index : \
                             start_index + frame_size].copy()

            # ハミング窓を掛ける
            frame = frame * np.hamming(frame_size)
          
            # 高速フーリエ変換(FFT)を実行
            spectrum = np.fft.fft(frame, n=fft_size)

            # 対数振幅スペクトルを計算
            log_absolute = np.log(np.abs(spectrum) + 1E-7)

            # 計算結果をスペクトログラムに格納
            spectrogram[frame_idx, :] = log_absolute

        #
        # 時間波形とスペクトログラムをプロット
        #

        # 描画領域を縦に2分割し、
        # スペクトログラムをプロットする
        plt.subplot(2, 1, n+1)

        # スペクトログラムの最大値を0に合わせて
        # カラーマップのレンジを調整
        spectrogram -= np.max(spectrogram)
        vmax = np.abs(np.min(spectrogram)) * 0.0
        vmin = - np.abs(np.min(spectrogram)) * 0.7

        # ヒストグラムをプロット
        plt.imshow(spectrogram.T[-1::-1,:], 
                   extent=[0, max_num_samples / sample_frequency, 
                           0, sample_frequency],
                   cmap = 'gray',
                   vmax = vmax,
                   vmin = vmin,
                   aspect = 'auto')
        plt.ylim([0, sample_frequency/2])

        # プロットのタイトルと、横軸と縦軸のラベルを定義
        plt.title('spectrogram')
        plt.xlabel('Time [sec]')
        plt.ylabel('Frequency [Hz]')

    # プロットを保存する
    plt.savefig(out_plot)

