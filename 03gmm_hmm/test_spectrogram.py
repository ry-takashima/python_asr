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
    wav_file = '../data/wav/BASIC5000_0001.wav'

    # フレームサイズ [ミリ秒]
    frame_size = 25
    # フレームシフト [ミリ秒]
    frame_shift = 10

    # プロットを出力するファイル(pngファイル)
    out_plot = './spectrogram.png'

    # wavファイルを開き、以降の処理を行う
    with wave.open(wav_file) as wav:
        # サンプリング周波数 [Hz] を取得
        sample_frequency = wav.getframerate()

        # wavデータのサンプル数を取得
        num_samples = wav.getnframes()

        # wavデータを読み込む
        waveform = wav.readframes(num_samples)

        # 読み込んだデータはバイナリ値(16bit integer)
        # なので，数値(整数)に変換する
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
    num_frames = (num_samples - frame_size) // frame_shift + 1

    # スペクトログラムの行列を用意
    spectrogram = np.zeros((num_frames, int(fft_size/2)+1))

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

        # 振幅スペクトルを得る
        absolute = np.abs(spectrum)

        # 振幅スペクトルは左右対称なので，左半分までのみを用いる
        absolute = absolute[:int(fft_size/2) + 1]

        # 対数を取り、対数振幅スペクトルを計算
        log_absolute = np.log(absolute + 1E-7)

        # 計算結果をスペクトログラムに格納
        spectrogram[frame_idx, :] = log_absolute

    #
    # 時間波形とスペクトログラムをプロット
    #

    # プロットの描画領域を作成
    plt.figure(figsize=(10,10))

    # 描画領域を縦に2分割し、
    # 上側に時間波形をプロットする
    plt.subplot(2, 1, 1)

    # 横軸(時間軸)を作成する
    time_axis = np.arange(num_samples) / sample_frequency
    
    # 時間波形のプロット
    plt.plot(time_axis, waveform)

    # プロットのタイトルと、横軸と縦軸のラベルを定義
    plt.title('waveform')
    plt.xlabel('Time [sec]')
    plt.ylabel('Value')

    # 横軸の表示領域を0から波形終了時刻に制限
    plt.xlim([0, num_samples / sample_frequency])

    # 2分割された描画領域の下側に
    # スペクトログラムをプロットする
    plt.subplot(2, 1, 2)
    
    # スペクトログラムの最大値を0に合わせて
    # カラーマップのレンジを調整
    spectrogram -= np.max(spectrogram)
    vmax = np.abs(np.min(spectrogram)) * 0.0
    vmin = - np.abs(np.min(spectrogram)) * 0.7

    # ヒストグラムをプロット
    plt.imshow(spectrogram.T[-1::-1,:], 
               extent=[0, num_samples / sample_frequency, 
                       0, sample_frequency / 2],
               cmap = 'gray',
               vmax = vmax,
               vmin = vmin,
               aspect = 'auto')

    # プロットのタイトルと、横軸と縦軸のラベルを定義
    plt.title('spectrogram')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')

    # プロットを保存する
    plt.savefig(out_plot)

