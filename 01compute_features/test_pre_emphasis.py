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

    # FFT(高速フーリエ変換)を行う範囲のサンプル数
    # 2のべき乗である必要がある
    fft_size = 1024

    # プロットを出力するファイル(pngファイル)
    out_plot = './pre_emphasis.png'

    # wavファイルを開き、以降の処理を行う
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
    
    frame_emp = np.convolve(frame,np.array([1.0, -0.97]), mode='same')
    # numpyの畳み込みでは0番目の要素が処理されない(window[i-1]が存在しないので)ため，
    # window[0-1]をwindow[0]で代用して処理する
    frame_emp[0] -= 0.97*frame_emp[0]

    h = np.zeros(fft_size)
    h[0] = 1.0
    h[1] = -0.97

    frame = frame * np.hamming(fft_size)
    frame_emp = frame_emp * np.hamming(fft_size)

    # FFTを実施する
    spectrum = np.fft.fft(frame)
    spectrum_emp = np.fft.fft(frame_emp)
    spectrum_h = np.fft.fft(h)

    # 振幅スペクトルを得る
    absolute = np.abs(spectrum)
    absolute_emp = np.abs(spectrum_emp)
    absolute_h = np.abs(spectrum_h)

    # 振幅スペクトルは左右対称なので，左半分までのみを用いる
    absolute = absolute[:np.int(fft_size/2) + 1]
    absolute_emp = absolute_emp[:np.int(fft_size/2) + 1]
    absolute_h = absolute_h[:np.int(fft_size/2) + 1]

    # 対数を取り、対数振幅スペクトルを計算
    log_absolute = np.log(absolute + 1E-7)
    log_absolute_emp = np.log(absolute_emp + 1E-7)
    log_absolute_h = np.log(absolute_h + 1E-7)

    #
    # 時間波形と対数振幅スペクトルをプロット
    #

    # プロットの描画領域を作成
    plt.figure(figsize=(10,10))

    # 2分割された描画領域の下側に
    # 対数振幅スペクトルをプロットする
    plt.subplot(3, 1, 1)

    # 横軸(周波数軸)を作成する
    freq_axis = np.arange(np.int(fft_size/2)+1) \
                * sampling_frequency / fft_size
    
    # 対数振幅スペクトルをプロット
    plt.plot(freq_axis, log_absolute, color='k')

    # プロットのタイトルと、横軸と縦軸のラベルを定義
    #plt.title('log-absolute spectrum without pre-emphasis (x)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Value')

    # 横軸の表示領域を0～最大周波数に制限
    plt.xlim([0, sampling_frequency / 2]) 
    plt.ylim([0,15])

    # 2分割された描画領域の下側に
    # 対数振幅スペクトルをプロットする
    plt.subplot(3, 1, 2)

    # 横軸(周波数軸)を作成する
    freq_axis = np.arange(np.int(fft_size/2)+1) \
                * sampling_frequency / fft_size
    
    # 対数振幅スペクトルをプロット
    plt.plot(freq_axis, log_absolute_emp, color='k')

    # プロットのタイトルと、横軸と縦軸のラベルを定義
    #plt.title('log-absolute spectrum with pre-emphasis (x_emp)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Value')

    # 横軸の表示領域を0～最大周波数に制限
    plt.xlim([0, sampling_frequency / 2]) 
    plt.ylim([0,15])

    plt.subplot(3, 1, 3)

    # 横軸(周波数軸)を作成する
    freq_axis = np.arange(np.int(fft_size/2)+1) \
                * sampling_frequency / fft_size
    
    # 対数振幅スペクトルをプロット
    plt.plot(freq_axis, log_absolute_emp - log_absolute, linestyle='dashed', color='k')
    plt.plot(freq_axis, log_absolute_h, color='k')

    # プロットのタイトルと、横軸と縦軸のラベルを定義
    #plt.title('log-absolute spectra of pre-emphasis filter (h) and (x_emp - x)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Value')

    # 横軸の表示領域を0～最大周波数に制限
    plt.xlim([0, sampling_frequency / 2]) 

    # プロットを保存する
    plt.savefig(out_plot)

