# -*- coding: utf-8 -*-

#
# Monophone-HMM の定義ファイルを作成します．
# 作成するHMMは left-to-right 型で
# 混合数は1です．対角共分散行列を仮定します．
#

# hmmfunc.pyからMonoPhoneHMMクラスをインポート
from hmmfunc import MonoPhoneHMM

# osモジュールをインポート
import os

#
# メイン関数
#
if __name__ == "__main__":

    # 音素リスト
    phone_list_file = \
        './exp/data/train_small/phone_list'

    # 各音素HMMの状態数
    num_states = 3

    # 入力特徴量の次元数
    # ここではMFCCを使用するため，
    # MFCCの次元数を入れる
    num_dims = 13

    # 自己ループ確率の初期値
    prob_loop = 0.7

    # 出力フォルダ
    out_dir = \
      './exp/model_%dstate_1mix' % (num_states)

    #
    # 処理ここから
    #
    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(out_dir, exist_ok=True)

    # 音素リストファイルを開き，phone_listに格納
    phone_list = []
    with open(phone_list_file, mode='r') as f:
        for line in f:
            # 音素リストファイルから音素を取得
            phone = line.split()[0]
            # 音素リストの末尾に加える
            phone_list.append(phone)

    # MonoPhoneHMMクラスを呼び出す
    hmm = MonoPhoneHMM()

    # HMMのプロトタイプを作成する
    hmm.make_proto(phone_list, num_states,
                   prob_loop, num_dims)

    # HMMのプロトタイプをjson形式で保存
    hmm.save_hmm(os.path.join(out_dir, 'hmmproto'))

