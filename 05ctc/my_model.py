# -*- coding: utf-8 -*-

#
# モデル構造を定義します
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch.nn as nn

# 作成したEncoderクラスをインポート
from encoder import Encoder

# 作成した初期化関数をインポート
from initialize import lecun_initialization

import numpy as np


class MyCTCModel(nn.Module):
    ''' CTCモデルの定義
    dim_in:            入力次元数
    dim_enc_hid:       エンコーダの隠れ層の次元数
    dim_enc_proj:      エンコーダのProjection層の次元数
                       (これがエンコーダの出力次元数になる)
    dim_out:           出力の次元数
    enc_num_layers:    エンコーダのレイヤー数
    enc_bidirectional: Trueにすると，エンコーダに
                       bidirectional RNNを用いる
    enc_sub_sample:    エンコーダにおいてレイヤーごとに設定する，
                       フレームの間引き率
    enc_rnn_type:      エンコーダRNNの種類．'LSTM'か'GRU'を選択する
    '''
    def __init__(self,
                 dim_in,
                 dim_enc_hid,
                 dim_enc_proj, 
                 dim_out,
                 enc_num_layers=2, 
                 enc_bidirectional=True,
                 enc_sub_sample=None, 
                 enc_rnn_type='LSTM'):
        super(MyCTCModel, self).__init__()

        # エンコーダを作成
        self.encoder = Encoder(dim_in=dim_in, 
                               dim_hidden=dim_enc_hid, 
                               dim_proj=dim_enc_proj, 
                               num_layers=enc_num_layers, 
                               bidirectional=enc_bidirectional, 
                               sub_sample=enc_sub_sample, 
                               rnn_type=enc_rnn_type)
        
        # 出力層
        # 出力層への入力 = Projection層の出力
        self.out = nn.Linear(in_features=dim_enc_proj,
                             out_features=dim_out)

        # LeCunのパラメータ初期化を実行
        lecun_initialization(self)


    def forward(self,
                input_sequence,
                input_lengths):
        ''' ネットワーク計算(forward処理)の関数
        input_sequence: 各発話の入力系列 [B x Tin x D]
        input_lengths:  各発話の系列長(フレーム数) [B]
          []の中はテンソルのサイズ
          B:    ミニバッチ内の発話数(ミニバッチサイズ)
          Tin:  入力テンソルの系列長(ゼロ埋め部分含む)
          D:    入力次元数(dim_in)
        '''
        # エンコーダに入力する
        enc_out, enc_lengths = self.encoder(input_sequence,
                                            input_lengths)

        # 出力層に入力する
        output = self.out(enc_out)

        # デコーダ出力と，エンコーダ出力系列長情報を出力する
        return output, enc_lengths

