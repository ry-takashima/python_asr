# -*- coding: utf-8 -*-

#
# Pytorchで用いるDatasetの定義
#

# PytorchのDatasetモジュールをインポート
from torch.utils.data import Dataset

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# sysモジュールをインポート
import sys


class SequenceDataset(Dataset):
    ''' ミニバッチデータを作成するクラス
        torch.utils.data.Datasetクラスを継承し，
        以下の関数を定義する
        __len__: 総サンプル数を出力する関数
        __getitem__: 1サンプルのデータを出力する関数
    feat_scp:  特徴量リストファイル
    label_scp: ラベルファイル
    feat_mean: 特徴量の平均値ベクトル
    feat_std:  特徴量の次元毎の標準偏差を並べたベクトル 
    pad_index: バッチ化の際にフレーム数を合わせる
               ためにpaddingする整数値
    splice:    前後(splice)フレームを特徴量を結合する
               splice=1とすると，前後1フレーム分結合
               するので次元数は3倍になる．
               splice=0の場合は何もしない
    '''
    def __init__(self, 
                 feat_scp, 
                 label_scp, 
                 feat_mean, 
                 feat_std,
                 pad_index=0,
                 splice=0):
        # 発話の数
        self.num_utts = 0
        # 各発話のID
        self.id_list = []
        # 各発話の特徴量ファイルへのパスを記したリスト
        self.feat_list = []
        # 各発話の特徴量フレーム数を記したリスト
        self.feat_len_list = []
        # 特徴量の平均値ベクトル
        self.feat_mean = feat_mean
        # 特徴量の標準偏差ベクトル
        self.feat_std = feat_std
        # 標準偏差のフロアリング
        # (0で割ることが発生しないようにするため)
        self.feat_std[self.feat_std<1E-10] = 1E-10
        # 特徴量の次元数
        self.feat_dim = \
            np.size(self.feat_mean)
        # 各発話のラベル
        self.label_list = []
        # 各発話のラベルの長さを記したリスト
        self.label_len_list = []
        # フレーム数の最大値
        self.max_feat_len = 0
        # ラベル長の最大値
        self.max_label_len = 0
        # フレーム埋めに用いる整数値
        self.pad_index = pad_index
        # splice:前後nフレームの特徴量を結合
        self.splice = splice

        # 特徴量リスト，ラベルを1行ずつ
        # 読み込みながら情報を取得する
        with open(feat_scp, mode='r') as file_f, \
             open(label_scp, mode='r') as file_l:
            for (line_feats, line_label) in zip(file_f, file_l):
                # 各行をスペースで区切り，
                # リスト型の変数にする
                parts_feats = line_feats.split()
                parts_label = line_label.split()

                # 発話ID(partsの0番目の要素)が特徴量と
                # ラベルで一致していなければエラー
                if parts_feats[0] != parts_label[0]:
                    sys.stderr.write('IDs of feat and '\
                        'label do not match.\n')
                    exit(1)

                # 発話IDをリストに追加
                self.id_list.append(parts_feats[0])
                # 特徴量ファイルのパスをリストに追加
                self.feat_list.append(parts_feats[1])
                # フレーム数をリストに追加
                feat_len = np.int64(parts_feats[2])
                self.feat_len_list.append(feat_len)

                # ラベル(番号で記載)をint型の
                # numpy arrayに変換
                label = np.int64(parts_label[1:])
                # ラベルリストに追加
                self.label_list.append(label)
                # ラベルの長さを追加
                self.label_len_list.append(len(label))

                # 発話数をカウント
                self.num_utts += 1

        # フレーム数の最大値を得る
        self.max_feat_len = \
            np.max(self.feat_len_list)
        # ラベル長の最大値を得る
        self.max_label_len = \
            np.max(self.label_len_list)

        # ラベルデータの長さを最大フレーム長に
        # 合わせるため，pad_indexの値で埋める
        for n in range(self.num_utts):
            # 埋めるフレームの数
            # = 最大フレーム数 - 自分のフレーム数
            pad_len = self.max_label_len \
                    - self.label_len_list[n]
            # pad_indexの値で埋める
            self.label_list[n] = \
                np.pad(self.label_list[n], 
                       [0, pad_len], 
                       mode='constant', 
                       constant_values=self.pad_index)

    def __len__(self):
        ''' 学習データの総サンプル数を返す関数
        本実装では発話単位でバッチを作成するため，
        総サンプル数=発話数である．
        '''
        return self.num_utts


    def __getitem__(self, idx):
        ''' サンプルデータを返す関数
        本実装では発話単位でバッチを作成するため，
        idx=発話番号である．
        '''
        # 特徴量系列のフレーム数
        feat_len = self.feat_len_list[idx]
        # ラベルの長さ
        label_len = self.label_len_list[idx]

        # 特徴量データを特徴量ファイルから読み込む
        feat = np.fromfile(self.feat_list[idx], 
                           dtype=np.float32)
        # フレーム数 x 次元数の配列に変形
        feat = feat.reshape(-1, self.feat_dim)

        # 平均と標準偏差を使って正規化(標準化)を行う
        feat = (feat - self.feat_mean) / self.feat_std

        # splicing: 前後 n フレームの特徴量を結合する
        org_feat = feat.copy()
        for n in range(-self.splice, self.splice+1):
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

        # 特徴量データのフレーム数を最大フレーム数に
        # 合わせるため，0で埋める
        pad_len = self.max_feat_len - feat_len
        feat = np.pad(feat,
                      [(0, pad_len), (0, 0)],
                      mode='constant',
                      constant_values=0)

        # ラベル
        label = self.label_list[idx]

        # 発話ID
        utt_id = self.id_list[idx]

        # 特徴量，ラベル，フレーム数，
        # ラベル長，発話IDを返す
        return (feat, 
               label,
               feat_len,
               label_len,
               utt_id)

