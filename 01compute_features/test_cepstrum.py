# -*- coding: utf-8 -*-

#
# ケプストラム分析により音声の
# フォルマント成分を抽出します．
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
    # 以下の時刻は音素"a"を発話している
    target_time = 0.73

    # FFT(高速フーリエ変換)を行う範囲のサンプル数
    # 2のべき乗である必要がある
    fft_size = 1024

    # ケプストラムの低次と高次の境目を決める次数
    cep_threshold = 33

    # プロットを出力するファイル(pngファイル)
    out_plot = './cepstrum.png'

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
    frame = waveform[target_index: 
                     target_index + fft_size].copy()
    
    # ハミング窓を掛ける
    frame = frame * np.hamming(fft_size)

    # FFTを実施する
    spectrum = np.fft.fft(frame)

    # 対数パワースペクトルを得る
    log_power = 2 * np.log(np.abs(spectrum) + 1E-7)

    # 対数パワースペクトルの逆フーリエ変換により
    # ケプストラムを得る
    cepstrum = np.fft.ifft(log_power)

    # ケプストラムの高次部分をゼロにする
    cepstrum_low = cepstrum.copy()
    cepstrum_low[(cep_threshold+1):-(cep_threshold)] = 0.0

    # 高域カットしたケプストラムを再度フーリエ変換し，
    # 対数パワースペクトルを計算
    log_power_ceplo = np.abs(np.fft.fft(cepstrum_low))

    # 逆に，低次をゼロにしたケプストラムを求める
    cepstrum_high = cepstrum - cepstrum_low
    # ただし，表示上のため，ゼロ次元目はカットしない
    cepstrum_high[0] = cepstrum[0]

    # 低域カットしたケプストラムを再度フーリエ変換し，
    # 対数パワースペクトルを計算
    log_power_cephi = np.abs(np.fft.fft(cepstrum_high))


    # プロットの描画領域を作成
    plt.figure(figsize=(18,10))
    
    # 対数パワースペクトルの横軸(周波数軸)を作成する
    freq_axis = np.arange(fft_size) \
                * sampling_frequency / fft_size
 
    # 3種類の対数パワースペクトルをプロット
    for n, log_pow in enumerate([log_power, 
                                 log_power_ceplo , 
                                 log_power_cephi]):
        # 描画領域を3行2列に分割し，1列目にプロット
        plt.subplot(3, 2, n*2+1)
        plt.plot(freq_axis, log_pow, color='k')

        # 横軸と縦軸のラベルを定義
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Value')

        # 表示領域を制限
        plt.xlim([0, sampling_frequency / 2]) 
        plt.ylim([0, 30])

    # ケプストラムの横軸(ケフレンシ軸=時間軸)を作成する
    qefr_axis = np.arange(fft_size) / sampling_frequency

    # 3種類のケプストラムをプロット
    for n, cepst in enumerate([cepstrum, 
                               cepstrum_low , 
                               cepstrum_high]):
        # 描画領域を3行2列に分割し，2列目にプロット
        plt.subplot(3, 2, n*2+2)
        # ケプストラムは実部をプロット
        # (虚部はほぼゼロである)
        plt.plot(qefr_axis, np.real(cepst), color='k')

        # 横軸と縦軸のラベルを定義
        plt.xlabel('QueFrency [sec]')
        plt.ylabel('Value')

        # 表示領域を制限
        plt.xlim([0, fft_size / (sampling_frequency * 2)]) 
        plt.ylim([-1.0, 2.0])

    # プロットを保存する
    plt.savefig(out_plot)
