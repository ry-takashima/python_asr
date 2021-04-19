# -*- coding: utf-8 -*-

#
# 混合数1(Single Gaussian Model)のHMMを学習します．
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

    # 学習元のHMMファイル
    base_hmm = './exp/model_3state_1mix/0.hmm'

    # 訓練データの特徴量リストのファイル
    feat_scp = \
        '../01compute_features/mfcc/train_small/feats.scp'

    # 訓練データのラベルファイル
    label_file = \
        './exp/data/train_small/text_int'

    # 更新回数
    num_iter = 10

    # 学習に用いる発話数
    # 実際は全ての発話を用いるが，時間がかかるため
    # このプログラムでは一部の発話のみを使っている
    num_utters = 50

    # 出力ディレクトリ
    out_dir = os.path.dirname(base_hmm)

    #
    # 処理ここから
    #

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(out_dir, exist_ok=True)

    # MonoPhoneHMMクラスを呼び出す
    hmm = MonoPhoneHMM()

    # 学習前のHMMを読み込む
    hmm.load_hmm(base_hmm)

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

    # 特徴量リストファイルを開き，
    # 発話ID毎の特徴量ファイルのパスを得る
    feat_list = {}
    with open(feat_scp, mode='r') as f:
        # 特徴量のパスをfeat_listに追加していく
        # このとき，学習に用いる発話数分だけ追加する
        # (全てのデータを学習に用いると時間がかかるため)
        for n, line in enumerate(f):
            if n >= num_utters:
                break
            # 0列目は発話ID
            utt = line.split()[0]
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
            # feat_fileに登録
            feat_list[utt] = ff


    #
    # 学習処理
    #
    # num_iterの数だけ更新を繰り返す
    for iter in range(num_iter):
        print('%d-th iterateion' % (iter+1))
        # 学習(1iteration分)
        hmm.train(feat_list, label_list)
       
        # HMMのプロトタイプをjson形式で保存
        out_hmm = os.path.join(out_dir, 
                               '%d.hmm' % (iter+1))
        # 学習したHMMを保存
        hmm.save_hmm(out_hmm)
        print('saved model: %s' % (out_hmm))

