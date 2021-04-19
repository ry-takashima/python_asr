# -*- coding: utf-8 -*-

#
# モデル構造を定義します
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch.nn as nn

# 作成したEncoder, Decoderクラスをインポート
from encoder import Encoder
from decoder import Decoder

# 作成した初期化関数をインポート
from initialize import lecun_initialization

import numpy as np


class MyMTLModel(nn.Module):
    ''' Attention RNN + CTC によるEnd-to-Endモデルの定義
    dim_in:            入力次元数
    dim_enc_hid:       エンコーダの隠れ層の次元数
    dim_enc_proj:      エンコーダのProjection層の次元数
                       (これがエンコーダの出力次元数になる)
    dim_dec_hid:       デコーダのRNNの次元数
    dim_out:           出力の次元数(sosとeosを含む全トークン数)
    dim_att:           Attention機構の次元数
    att_filter_size:   LocationAwareAttentionのフィルタサイズ
    att_filter_num:    LocationAwareAttentionのフィルタ数
    sos_id:            <sos>トークンの番号
    enc_bidirectional: Trueにすると，エンコーダに
                       bidirectional RNNを用いる
    enc_sub_sample:    エンコーダにおいてレイヤーごとに設定する，
                       フレームの間引き率
    enc_rnn_type:      エンコーダRNNの種類．'LSTM'か'GRU'を選択する
    '''
    def __init__(self, dim_in, dim_enc_hid, dim_enc_proj, 
                 dim_dec_hid, dim_out, dim_att, 
                 att_filter_size, att_filter_num,
                 sos_id, att_temperature=1.0,
                 enc_num_layers=2, dec_num_layers=2, 
                 enc_bidirectional=True, enc_sub_sample=None, 
                 enc_rnn_type='LSTM'):
        super(MyMTLModel, self).__init__()

        # エンコーダを作成
        self.encoder = Encoder(dim_in=dim_in, 
                               dim_hidden=dim_enc_hid, 
                               dim_proj=dim_enc_proj, 
                               num_layers=enc_num_layers, 
                               bidirectional=enc_bidirectional, 
                               sub_sample=enc_sub_sample, 
                               rnn_type=enc_rnn_type)
        
        # デコーダを作成
        self.decoder = Decoder(dim_in=dim_enc_proj, 
                               dim_hidden=dim_dec_hid, 
                               dim_out=dim_out, 
                               dim_att=dim_att, 
                               att_filter_size=att_filter_size, 
                               att_filter_num=att_filter_num, 
                               sos_id=sos_id, 
                               att_temperature=att_temperature,
                               num_layers=dec_num_layers)

        # CTC出力層
        # 出力次元数はdim_outから<sos>分の1を引いた値
        self.ctc_out = nn.Linear(in_features=dim_enc_proj,
                                 out_features=dim_out)

        # LeCunのパラメータ初期化を実行
        lecun_initialization(self)


    def forward(self,
                input_sequence,
                input_lengths,
                label_sequence=None):
        ''' ネットワーク計算(forward処理)の関数
        input_sequence: 各発話の入力系列 [B x Tin x D]
        input_lengths:  各発話の系列長(フレーム数) [B]
        label_sequence: 各発話の正解ラベル系列(学習時に用いる) [B x Tout]
          []の中はテンソルのサイズ
          B:    ミニバッチ内の発話数(ミニバッチサイズ)
          Tin:  入力テンソルの系列長(ゼロ埋め部分含む)
          D:    入力次元数(dim_in)
          Tout: 正解ラベル系列の系列長(ゼロ埋め部分含む)
        '''
        # エンコーダに入力する
        enc_out, enc_lengths = self.encoder(input_sequence,
                                            input_lengths)
 
        # CTC出力層にエンコーダ出力を入れる
        ctc_output = self.ctc_out(enc_out)

        # デコーダに入力する
        dec_out = self.decoder(enc_out,
                               enc_lengths,
                               label_sequence)
        # デコーダ出力と，エンコーダ出力、
        # エンコーダ出力系列長情報を出力する
        return dec_out, ctc_output, enc_lengths


    def save_att_matrix(self, utt, filename):
        ''' Attention行列を画像にして保存する
        utt:      出力する、バッチ内の発話番号
        filename: 出力ファイル名
        '''
        # decoderのsave_att_matrixを実行
        self.decoder.save_att_matrix(utt, filename)

