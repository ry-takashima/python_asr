# -*- coding: utf-8 -*-

#
# HMMのプロトタイプを読み込み，
# フラットスタートで初期化します．
#

# hmmfunc.pyからMonoPhoneHMMクラスをインポート
from hmmfunc import MonoPhoneHMM

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# osモジュールをインポート
import os

#
# メイン関数
#
if __name__ == "__main__":

    # HMMプロトタイプ
    hmmproto = './exp/model_3state_1mix/hmmproto'

    # 学習データの特徴量の平均/標準偏差ファイル
    mean_std_file = \
        '../01compute_features/mfcc/train_small/mean_std.txt'

    # 出力ディレクトリ
    out_dir = os.path.dirname(hmmproto)

    #
    # 処理ここから
    #

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(out_dir, exist_ok=True)

    # 特徴量の平均/標準偏差ファイルを読み込む
    with open(mean_std_file, mode='r') as f:
        # 全行読み込み
        lines = f.readlines()
        # 1行目(0始まり)が平均値ベクトル(mean)，
        # 3行目が標準偏差ベクトル(std)
        mean_line = lines[1]
        std_line = lines[3]
        # スペース区切りのリストに変換
        mean = mean_line.split()
        std = std_line.split()
        # numpy arrayに変換
        mean = np.array(mean, dtype=np.float64)
        std = np.array(std, dtype=np.float64)
        # 標準偏差を分散に変換
        var = std ** 2
    
    # MonoPhoneHMMクラスを呼び出す
    hmm = MonoPhoneHMM()

    # HMMプロトタイプを読み込む
    hmm.load_hmm(hmmproto)

    # フラットスタート初期化を実行
    hmm.flat_init(mean, var)

    # HMMのプロトタイプをjson形式で保存
    hmm.save_hmm(os.path.join(out_dir, '0.hmm'))

