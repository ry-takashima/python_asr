# -*- coding: utf-8 -*-

#
# 訓練データと開発データの
# HMM状態レベルでのアライメントを推定します
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

    # 訓練データの特徴量リストのファイル
    train_feat_scp = \
        '../01compute_features/mfcc/train_small/feats.scp'
    # 開発データの特徴量リストのファイル
    dev_feat_scp = \
        '../01compute_features/mfcc/dev/feats.scp'

    # 訓練データのラベルファイル
    train_label_file = \
        '../03gmm_hmm/exp/data/train_small/text_int'
    # 開発データのラベルファイル
    dev_label_file = \
        '../03gmm_hmm/exp/data/dev/text_int'

    # 訓練データのアライメント結果の出力ファイル
    train_align_file = \
        './exp/data/train_small/alignment'
    # 開発データのアライメント結果の出力ファイル
    dev_align_file = \
        './exp/data/dev/alignment'

    #
    # 処理ここから
    #

    # MonoPhoneHMMクラスを呼び出す
    hmm = MonoPhoneHMM()

    # 学習前のHMMを読み込む
    hmm.load_hmm(hmm_file)

    # 訓練/開発データの
    # 特徴量/ラベル/アライメントファイル
    # をリスト化
    feat_scp_list = [train_feat_scp, dev_feat_scp]
    label_file_list = [train_label_file, dev_label_file]
    align_file_list = [train_align_file, dev_align_file]

    for feat_scp, label_file, align_file in \
          zip(feat_scp_list, label_file_list, align_file_list):

        # 出力ディレクトリ
        out_dir = os.path.dirname(align_file)

        # 出力ディレクトリが存在しない場合は作成する
        os.makedirs(out_dir, exist_ok=True)

        # ラベルファイルを開き，発話ID毎の
        # ラベル情報を得る
        label_list = {}
        with open(label_file, mode='r') as f:
            for line in f:
                # 0列目は発話ID
                utt = line.split()[0]
                # 1列目以降はラベル
                lab = line.split()[1:]
                # 各要素は文字として読み込まれているので，
                # 整数値に変換する
                lab = np.int64(lab)
                # label_listに登録
                label_list[utt] = lab

        # 発話毎にアライメントを推定する
        with open(align_file, mode='w') as fa, \
             open(feat_scp, mode='r') as fs:
            for line in fs:
                # 0列目は発話ID
                utt = line.split()[0]
                print(utt)
                # 1列目はファイルパス
                ff = line.split()[1]
                # 3列目は次元数
                nd = int(line.split()[3])

                # 発話IDがlabel_に存在しなければエラー
                if not utt in label_list:
                    sys.stderr.write(\
                        '%s does not have label\n' % (utt))
                    exit(1)
                # 次元数がHMMの次元数と一致しなければエラー
                if hmm.num_dims != nd:
                    sys.stderr.write(\
                        '%s: unexpected #dims (%d)\n'\
                        % (utt, nd))
                    exit(1)

                # ラベルを得る
                label = label_list[utt]
                # 特徴量ファイルを開く
                feat = np.fromfile(ff, dtype=np.float32)
                # フレーム数 x 次元数の配列に変形
                feat = feat.reshape(-1, hmm.num_dims)
                
                # アライメントの実行
                alignment = hmm.state_alignment(feat, label)
                # alignmentは数値のリストなので
                # ファイルに書き込むために文字列に変換する
                alignment = ' '.join(map(str, alignment))
                # ファイルに出力する
                fa.write('%s %s\n' % (utt, alignment))

