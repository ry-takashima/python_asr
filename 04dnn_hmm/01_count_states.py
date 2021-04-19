# -*- coding: utf-8 -*-

#
# 推定されたアライメント結果をもとに，
# 各HMM状態の出現回数をカウントします．
#

# hmmfunc.pyからMonoPhoneHMMクラスをインポート
from hmmfunc import MonoPhoneHMM

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# os, sysモジュールをインポート
import sys
import os

#
# メイン関数
#
if __name__ == "__main__":

    # HMMファイル
    hmm_file = '../03gmm_hmm/exp/model_3state_2mix/10.hmm'

    # 訓練データのアライメントファイル
    align_file = \
        './exp/data/train_small/alignment'

    # 計算した事前確率ファイル
    count_file = \
        './exp/model_dnn/state_counts'

    #
    # 処理ここから
    #

    # MonoPhoneHMMクラスを呼び出す
    hmm = MonoPhoneHMM()

    # 学習前のHMMを読み込む
    hmm.load_hmm(hmm_file)

    # HMMの総状態数を得る
    num_states = hmm.num_phones * hmm.num_states

    # 出力ディレクトリ
    out_dir = os.path.dirname(count_file)

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(out_dir, exist_ok=True)

    # 状態毎の出現カウンタ
    count = np.zeros(num_states, np.int64)

    # アライメントファイルを開く
    with open(align_file, mode='r') as f:
        for line in f:
            # 0列目は発話ID
            utt = line.split()[0]
            # 1列目以降はアライメント
            ali = line.split()[1:]
            # アライメントは文字として
            # 読み込まれているので，
            # 整数値に変換する
            ali = np.int64(ali)
            # 状態を一つずつ読み込み，
            # 該当する状態のカウンタを1増やす
            for a in ali:
                count[a] += 1

    # カウントが0のものは1にする
    # 以降の処理でゼロ割りの発生を防ぐため
    count[count==0] = 1

    # カウント結果を出力する
    with open(count_file, mode='w') as f:
        # ベクトルcountを文字列に変換する
        count_str = ' '.join(map(str, count))
        f.write('%s\n' % (count_str))

