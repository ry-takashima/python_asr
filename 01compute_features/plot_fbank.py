# -*- coding: utf-8 -*-

#
# メルフィルタバンクを作成してプロットします．
#

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# プロット用モジュール(matplotlib)をインポート
import matplotlib.pyplot as plt

#
# 周波数をヘルツからメルに変換する
#
def Herz2Mel(herz):
    return (1127.0 * np.log(1.0 + herz / 700))

#
# メイン関数
#
if __name__ == "__main__":

    # 最大周波数 [Hz]
    max_herz = 8000

    # FFTのポイント数
    fft_size = 4096

    # フィルタバンクの数
    num_mel_bins = 7

    # プロットを出力するファイル(pngファイル)
    out_plot = './mel_bank.png'

    # メル軸での最大周波数
    max_mel = Herz2Mel(max_herz)
    
    # メル軸上での等間隔な周波数を得る
    mel_points = np.linspace(0, max_mel, num_mel_bins+2)

    # パワースペクトルの次元数 = FFTサイズ/2+1
    dim_spectrum = int(fft_size / 2) + 1

    # メルフィルタバンク(フィルタの数 x スペクトルの次元数)
    mel_filter_bank = np.zeros((num_mel_bins, dim_spectrum))
    for m in range(num_mel_bins):
        # 三角フィルタの左端，中央，右端のメル周波数
        left_mel = mel_points[m]
        center_mel = mel_points[m+1]
        right_mel = mel_points[m+2]
        # パワースペクトルの各ビンに対応する重みを計算する
        for n in range(dim_spectrum):
            # 各ビンに対応するヘルツ軸周波数を計算
            freq = 1.0 * n * max_herz / dim_spectrum
            # メル周波数に変換
            mel = Herz2Mel(freq)
            # そのビンが三角フィルタの範囲に入っていれば，重みを計算
            if mel > left_mel and mel < right_mel:
                if mel <= center_mel:
                    weight = (mel - left_mel) / (center_mel - left_mel)
                else:
                    weight = (right_mel-mel) / (right_mel-center_mel)
                mel_filter_bank[m][n] = weight
 
    # プロットの描画領域を作成
    plt.figure(figsize=(6,4))

    # 横軸(周波数軸)を作成する
    freq_axis = np.arange(dim_spectrum) \
                * max_herz / dim_spectrum
    
    for m in range(num_mel_bins):
        # フィルタバンクをプロット
        plt.plot(freq_axis, mel_filter_bank[m], color='k')
        #plt.plot(freq_axis, mel_filter_bank[m])

    plt.xlabel('Frequency [Hz]')

    # 横軸の表示領域を0～最大周波数に制限
    plt.xlim([0, max_herz]) 

    # プロットを保存する
    plt.savefig(out_plot)

