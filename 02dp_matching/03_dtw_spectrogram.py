# -*- coding: utf-8 -*-

#
# 音声のスペクトログラムを作成後，
# アライメントに従ってスペクトログラムを
# 引き延ばして描画する
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

    alignment_file = './alignment.txt'

    # サンプリング周波数
    sample_frequency = 16000
    # フレームサイズ [ミリ秒]
    frame_size = 25
    # フレームシフト [ミリ秒]
    frame_shift = 10

    # プロットを出力するファイル(pngファイル)
    out_plot = './dtw_spectrogram.png'

    # フレームサイズをミリ秒からサンプル数に変換
    frame_size = int(sample_frequency * frame_size * 0.001)

    # フレームシフトをミリ秒からサンプル数へ変換
    frame_shift = int(sample_frequency * frame_shift * 0.001)

    # FFTを行う範囲のサンプル数を，
    # フレームサイズ以上の2のべき乗に設定
    fft_size = 1
    while fft_size < frame_size:
        fft_size *= 2

    # アライメント情報を得る
    alignment = []
    with open(alignment_file, mode='r') as f:
        for line in f:
            parts = line.split()
            alignment.append([int(parts[0]), int(parts[1])])

    # プロットの描画領域を作成
    plt.figure(figsize=(10,10))

    # 2個のwavfileに対して以下を実行
    for file_id, wav_file in enumerate([wav_file_1, wav_file_2]):
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
        spectrogram = np.zeros((num_frames, fft_size))

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

        # スペクトログラムを
        # アライメントに合わせて拡大する
        dtw_spectrogram = np.zeros((len(alignment), fft_size))
        for t in range(len(alignment)):
            # 対応するフレーム番号
            idx = alignment[t][file_id]
            # 対応するフレーム番号のスペクトログラムを
            # コピーして引き延ばす
            dtw_spectrogram[t, :] = spectrogram[idx, :]

        #
        # 時間波形とスペクトログラムをプロット
        #

        # 描画領域を縦に2分割し、
        # スペクトログラムをプロットする
        plt.subplot(2, 1, file_id+1)

        # スペクトログラムの最大値を0に合わせて
        # カラーマップのレンジを調整
        dtw_spectrogram -= np.max(dtw_spectrogram)
        vmax = np.abs(np.min(dtw_spectrogram)) * 0.0
        vmin = - np.abs(np.min(dtw_spectrogram)) * 0.7

        # ヒストグラムをプロット
        plt.imshow(dtw_spectrogram.T[-1::-1,:], 
                   extent=[0, len(alignment) * \
                              frame_shift / sample_frequency, 
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

