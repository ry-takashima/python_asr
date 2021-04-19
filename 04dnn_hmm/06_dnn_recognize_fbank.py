# -*- coding: utf-8 -*-

#
# DNN-HMMで認識します．
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn.functional as F

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# hmmfunc.pyからMonoPhoneHMMクラスをインポート
from hmmfunc import MonoPhoneHMM

# 作成したDatasetクラスをインポート
from my_dataset import SequenceDataset

# モデルの定義をインポート
from my_model import MyDNN

# json形式の入出力を行うモジュールをインポート
import json

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

    # 評価データの特徴量リスト
    test_feat_scp = \
        './exp/data/test/feats.scp'
    
    # HMMファイル
    hmm_file = '../03gmm_hmm/exp/model_3state_2mix/10.hmm'
 
    # DNNモデルファイル
    dnn_file = './exp/model_dnn_fbank/best_model.pt'

    # HMM状態出現カウントファイル
    count_file = './exp/model_dnn/state_counts'

    # 訓練データから計算された
    # 特徴量の平均/標準偏差ファイル
    mean_std_file = 'exp/model_dnn_fbank/mean_std.txt'

    # 辞書ファイル
    lexicon_file = '../03gmm_hmm/exp/data/test/lexicon.txt'

    # 音素リスト
    phone_list_file = \
        '../03gmm_hmm/exp/data/train_small/phone_list'

    # Trueの場合，文章の先頭と末尾に
    # ポーズがあることを前提とする
    insert_sil = True

    # DNN学習時に出力した設定ファイル
    config_file = os.path.join(\
                      os.path.dirname(dnn_file),
                      'config.json')

    #
    # 設定ここまで
    #

    # MonoPhoneHMMクラスを呼び出す
    hmm = MonoPhoneHMM()
    # HMMを読み込む
    hmm.load_hmm(hmm_file)

    # 設定ファイルを読み込む
    with open(config_file, mode='r') as f:
        config = json.load(f)

    # 読み込んだ設定を反映する
    # 中間層のレイヤー数
    num_layers = config['num_layers']
    # 中間層の次元数
    hidden_dim = config['hidden_dim']
    # spliceフレーム数
    splice = config['splice']

    # 特徴量の平均/標準偏差ファイルを読み込む
    with open(mean_std_file, mode='r') as f:
        # 全行読み込み
        lines = f.readlines()
        # 1行目(0始まり)が平均値ベクトル(mean)，
        # 3行目が標準偏差ベクトル(std)
        mean_line = lines[1]
        std_line = lines[3]
        # スペース区切りのリストに変換
        feat_mean = mean_line.split()
        feat_std = std_line.split()
        # numpy arrayに変換
        feat_mean = np.array(feat_mean, 
                                dtype=np.float32)
        feat_std = np.array(feat_std, 
                               dtype=np.float32)

    # 次元数の情報を得る
    feat_dim = np.size(feat_mean)

    # DNNの出力層の次元数は音素数x状態数
    dim_out = hmm.num_phones * hmm.num_states

    # ニューラルネットワークモデルを作成する
    # 入力特徴量の次元数は
    # feat_dim * (2*splice+1)
    dim_in = feat_dim * (2*splice+1)
    model = MyDNN(dim_in=dim_in,
                  dim_hidden=hidden_dim,
                  dim_out=dim_out, 
                  num_layers=num_layers)
    
    # 学習済みのDNNファイルから
    # モデルのパラメータを読み込む
    model.load_state_dict(torch.load(dnn_file))

    # モデルを評価モードに設定する
    model.eval()

    # HMM状態カウントファイルを読み込む
    with open(count_file, mode='r') as f:
        # 1行読み込む
        line = f.readline()
        # HMM状態毎の出現回数がスペース区切りで
        # 入っているので，スペースで区切って
        # リスト化する
        count = line.split()
        # 各数値は文字型になっているので
        # 数値に変換
        count = np.float32(count)

        # 総和で割ることで，各HMM状態の
        # 事前発生確率に変換
        prior = count / np.sum(count)

    # 音素リストファイルを開き，phone_listに格納
    phone_list = []
    with open(phone_list_file, mode='r') as f:
        for line in f:
            # 音素リストファイルから音素を取得
            phone = line.split()[0]
            # 音素リストの末尾に加える
            phone_list.append(phone)

    # 辞書ファイルを開き，単語と音素列の対応リストを得る
    lexicon = []
    with open(lexicon_file, mode='r') as f:
        for line in f:
            # 0列目は単語
            word = line.split()[0]
            # 1列目以降は音素列
            phones = line.split()[1:]
            # insert_silがTrueの場合は両端にポーズを追加
            if insert_sil:
                phones.insert(0, phone_list[0])
                phones.append(phone_list[0])
            # phone_listを使って音素を数値に変換
            ph_int = []
            for ph in phones:
                if ph in phone_list:
                    ph_int.append(phone_list.index(ph))
                else:
                    sys.stderr.write('invalid phone %s' % (ph))
            # 単語,音素列,数値表記の辞書として
            # lexiconに追加
            lexicon.append({'word': word,
                            'pron': phones,
                            'int': ph_int})

    #
    # ここから認識処理:
    #

    # 特徴量リストファイルを開く
    with open(test_feat_scp, mode='r') as f:
        for line in f:
            # 0列目は発話ID
            utt = line.split()[0]
            # 1列目はファイルパス
            ff = line.split()[1]
            # 3列目は次元数
            nd = int(line.split()[3])
            
            # 特徴量データを読み込む
            feat = np.fromfile(ff, dtype=np.float32)
            # フレーム数 x 次元数の配列に変形
            feat = feat.reshape(-1, nd)

            # 平均と標準偏差を使って正規化(標準化)を行う
            feat = (feat - feat_mean) / feat_std

            # splicing: 前後 n フレームの特徴量を結合する
            org_feat = feat.copy()
            for n in range(-splice, splice+1):
                # 元々の特徴量を n フレームずらす
                tmp = np.roll(org_feat, n, axis=0)
                if n < 0:
                    # 前にずらした場合は
                    # 終端nフレームを0にする
                    tmp[n:] = 0
                elif n > 0:
                    # 後ろにずらした場合は
                    # 始端nフレームを0にする
                    tmp[:n] = 0
                else:
                    continue
                # ずらした特徴量を次元方向に
                # 結合する
                feat = np.hstack([feat,tmp])

            # pytorchのDNNに入力するため，
            # torch.tensor型に変換する
            feat = torch.tensor(feat)
            
            # DNNに入力する
            output = model(feat)

            # softmax関数に入れて確率に変換する
            output = F.softmax(output, dim=1)
            
            # numpy array型に戻す
            output = output.detach().numpy()

            # 各HMM状態の事前発生確率で割り，
            # さらに対数を取って対数尤度に変換する
            # (HMMの各状態の出力確率に相当する形にする)
            likelihood = np.log(output / prior)

            (result, detail) = hmm.recognize_with_dnn(likelihood, lexicon)
            # 結果を出力する
            sys.stdout.write('%s %s\n' % (utt, ff))
            sys.stdout.write('Result = %s\n' % (result))
            sys.stdout.write('[Runking]\n')
            for res in detail:
                sys.stdout.write('  %s %f\n' \
                    % (res['word'], res['score']))

