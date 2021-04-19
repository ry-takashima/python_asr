# -*- coding: utf-8 -*-

#
# wavファイルの一部の区間をフーリエ変換し，
# 振幅スペクトルをプロットします．
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

    # 分析する時刻．BASIC5000_0001.wav では，
    # 以下の時刻は音素"o"を発話している
    target_time = 0.58
    target_time = 0.74

    # FFT(高速フーリエ変換)を行う範囲のサンプル数
    # 2のべき乗である必要がある
    fft_size = 1024

    # プロットを出力するファイル(pngファイル)
    out_plot = './spectrum.png'

    # wavファイルを開き，以降の処理を行う
    with wave.open(wav_file) as wav:
        # サンプリング周波数 [Hz] を取得
        sampling_frequency = wav.getframerate()

        # wavデータを読み込む
        waveform = wav.readframes(wav.getnframes())

        # 読み込んだデータはバイナリ値(16bit integer)
        # なので，数値(整数)に変換する
        waveform = np.frombuffer(waveform, dtype=np.int16)

    # 分析する時刻をサンプル番号に変換
    target_index = np.int(target_time * sampling_frequency)

    # FFTを実施する区間分の波形データを取り出す
    frame = waveform[target_index: target_index + fft_size]
    
    # FFTを実施する
    spectrum = np.fft.fft(frame)

    # 振幅スペクトルを得る
    absolute = np.abs(spectrum)

    # 振幅スペクトルは左右対称なので，左半分までのみを用いる
    absolute = absolute[:np.int(fft_size/2) + 1]

    # 対数を取り，対数振幅スペクトルを計算
    log_absolute = np.log(absolute + 1E-7)

    #
    # 時間波形と対数振幅スペクトルをプロット
    #

    # プロットの描画領域を作成
    plt.figure(figsize=(10,10))

    # 描画領域を縦に2分割し，
    # 上側に時間波形をプロットする
    plt.subplot(2, 1, 1)

    # 横軸(時間軸)を作成する
    time_axis = target_time \
                + np.arange(fft_size) / sampling_frequency
    
    # 時間波形のプロット
    plt.plot(time_axis, frame)

    # プロットのタイトルと，横軸と縦軸のラベルを定義
    plt.title('waveform')
    plt.xlabel('Time [sec]')
    plt.ylabel('Value')

    # 横軸の表示領域を分析区間の時刻に制限
    plt.xlim([target_time, 
              target_time + fft_size / sampling_frequency])

    # 2分割された描画領域の下側に
    # 対数振幅スペクトルをプロットする
    plt.subplot(2, 1, 2)

    # 横軸(周波数軸)を作成する
    freq_axis = np.arange(np.int(fft_size/2)+1) \
                * sampling_frequency / fft_size
    
    # 対数振幅スペクトルをプロット
    plt.plot(freq_axis, log_absolute)

    # プロットのタイトルと，横軸と縦軸のラベルを定義
    plt.title('log-absolute spectrum')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Value')

    # 横軸の表示領域を0～最大周波数に制限
    plt.xlim([0, sampling_frequency / 2]) 

    # プロットを保存する
    plt.savefig(out_plot)

