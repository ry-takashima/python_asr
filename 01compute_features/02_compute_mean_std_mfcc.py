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
  
  #
  # 設定ここから
  #

  # 各特徴量ファイルのリストと平均・標準偏差計算結果の出力先
  train_small_feat_scp = './mfcc/train_small/feats.scp'
  train_small_out_dir = './mfcc/train_small'
  train_large_feat_scp = './mfcc/train_large/feats.scp'
  train_large_out_dir = './mfcc/train_large'

  # 特徴量ファイルリストと出力先をリストにする
  feat_scp_list = [train_small_feat_scp, train_large_feat_scp]
  out_dir_list = [train_small_out_dir, train_large_out_dir]

  # 各セットについて処理を実行する
  for (feat_scp, out_dir) in zip(feat_scp_list, out_dir_list):
    print('Input feat_scp: %s' % (feat_scp))

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(out_dir, exist_ok=True)

    # 特徴量の平均と分散
    feat_mean = None
    feat_var = None
    # 総フレーム数
    total_frames = 0

    # 特徴量リストを開く
    with open(feat_scp, mode='r') as file_feat:
      # 特徴量リストを1行ずつ読み込む
      for i, line in enumerate(file_feat):
        # 各行には，発話ID，特徴量ファイルのパス，フレーム数，次元数がスペース区切りで記載されている
        # split関数を使ってスペース区切りの行をリスト型の変数に変換する
        parts = line.split()
        # 0番目が発話ID
        utterance_id = parts[0]
        # 1番目が特徴量ファイルのパス
        feat_path = parts[1]
        # 2番目がフレーム数
        num_frames = int(parts[2])
        # 3番目が次元数
        num_dims = int(parts[3])
               
        # 特徴量データを特徴量ファイルから読み込む
        feature = np.fromfile(feat_path, dtype=np.float32)

        # 読み込んだ時点で，featureは1行のベクトル(要素数=フレーム数*次元数)として格納されている
        # これをフレーム数 x 次元数の行列形式に変換する
        feature = feature.reshape(num_frames, num_dims)
 
        # 最初のファイルを処理した時に，平均と分散を初期化
        if i == 0:
          feat_mean = np.zeros(num_dims, np.float32)
          feat_var = np.zeros(num_dims, np.float32)

        # 総フレーム数を加算
        total_frames += num_frames
        # 特徴量ベクトルのフレーム総和を加算
        feat_mean += np.sum(feature, axis=0)
        # 特徴量ベクトルの二乗のフレーム総和を加算
        feat_var += np.sum(np.power(feature,2), axis=0)
    
    # 総フレーム数で割って平均値ベクトルを計算
    feat_mean /= total_frames
    # 分散値ベクトルを計算
    feat_var = (feat_var / total_frames) - np.power(feat_mean,2)
    # 平方根を取って標準偏差ベクトルを算出
    feat_std = np.sqrt(feat_var)

    # ファイルに書き込む
    out_file = os.path.join(out_dir, 'mean_std.txt')
    print('Output file: %s' % (out_file))
    with open(out_file, mode='w') as file_o:
      # 平均値ベクトルの書き込み
      file_o.write('mean\n')
      for i in range(np.size(feat_mean)):
        file_o.write('%e ' % (feat_mean[i]))
      file_o.write('\n')
      # 標準偏差ベクトルの書き込み
      file_o.write('std\n')
      for i in range(np.size(feat_std)):
        file_o.write('%e ' % (feat_std[i]))
      file_o.write('\n')

