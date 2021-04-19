# -*- coding: utf-8 -*-

#
# 学習データから，特徴量の平均と標準偏差を求めます．
#

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# os, sysモジュールをインポート
import os
import sys

#
# メイン関数
#
if __name__ == "__main__":
  
  mstd_file = './fbank/train_small/mean_std.txt'

  with open(mstd_file, mode='r') as file_m:
    # 全行読み込み
    lines = file_m.readlines()
    # 1行目(0始まり)がmean
    mean_line = lines[1]
    # 3行目がstd
    std_line = lines[3]
    # スペース区切りのリストに変換
    feat_mean = mean_line.split()
    # numpy arrayに変換(dtypeを使って文字列からfloatに変換する点に注意)
    feat_mean = np.array(feat_mean, dtype=np.float32)
    print(np.shape(feat_mean))
    print(feat_mean)
