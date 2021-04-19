# -*- coding: utf-8 -*-

#
# wavファイルを開いて波形をプロットします．
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

    # 波形のプロットを出力するファイル(pngファイル)
    out_plot = './plot.png'

    # wavファイルを開き、以降の処理を行う
    with wave.open(wav_file) as wav:
        # サンプリング周波数 [Hz] を取得
        sampling_frequency = wav.getframerate()

        # サンプルサイズ [Byte] を取得
        sample_size = wav.getsampwidth()

        # チャネル数を取得
        num_channels = wav.getnchannels()

        # wavデータのサンプル数を取得
        num_samples = wav.getnframes()

        # wavデータを読み込む
        waveform = wav.readframes(num_samples)

        # 読み込んだデータはバイナリ値(16bit integer)
        # なので，数値(整数)に変換する
        waveform = np.frombuffer(waveform, dtype=np.int16)

    #
    # 読み込んだwavファイルの情報を表示する
    #
    print("Sampling Frequency: %d [Hz]" % sampling_frequency)
    print("Sample Size: %d [Byte]" % sample_size)
    print("Number of Channels: %d" % num_channels)
    print("Number of Samples: %d" % num_samples)

    #
    # 読み込んだ波形(waveform)をプロットする
    #
    
    # 横軸(時間軸)を作成する
    time_axis = np.arange(num_samples) / sampling_frequency

    # プロットの描画領域を作成
    plt.figure(figsize=(10,4))

    # プロット
    plt.plot(time_axis, waveform)

    # 横軸と縦軸のラベルを定義
    plt.xlabel("Time [sec]")
    plt.ylabel("Value")

    # 横軸の表示領域を0から波形終了時刻に制限
    plt.xlim([0, num_samples / sampling_frequency])

    # プロットを保存する
    plt.savefig(out_plot)

