# -*- coding: utf-8 -*-

#
# RNN エンコーダ部の実装です．
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class Encoder(nn.Module):
    ''' エンコーダ
    dim_in:        入力特徴量の次元数
    dim_hidden:    隠れ層の次元数(bidirectional=Trueの場合，
                   実際の次元数はdim_hidden * 2)
    dim_proj:      Projection層の次元数
                   (これがエンコーダの出力次元数になる)
    num_layers:    RNN層(およびProjection層)の数
    bidirectional: Trueにすると，bidirectional RNNを用いる
    sub_sample:    レイヤーごとに設定する，フレームの間引き率
                   num_layers=4のとき，sub_sample=[1,2,3,1]とすると
                   2層目でフレーム数を1/2に，3層目で1/3にする
                   (出力のフレーム数は1/6になる)
    rnn_type:      'LSTM'か'GRU'を選択する
    '''
    def __init__(self,
                 dim_in,
                 dim_hidden,
                 dim_proj,
                 num_layers=2, 
                 bidirectional=True,
                 sub_sample=None,
                 rnn_type='LSTM'):
        super(Encoder, self).__init__()
        # RNN層の数
        self.num_layers = num_layers

        # RNN層は1層ずつ定義して，リスト化する
        rnn = []
        for n in range(self.num_layers):
            # RNNへの入力次元数は，
            # 最初の層のみdim_in，それ以外はdim_proj
            input_size = dim_in if n == 0 else dim_proj
            # rnn_type がGRUならGRUを，それ以外ならLSTMを用いる
            if rnn_type == 'GRU':
                rnn.append(nn.GRU(input_size=input_size, 
                                  hidden_size=dim_hidden,
                                  num_layers=1, 
                                  bidirectional=bidirectional, 
                                  batch_first=True))
            else:
                rnn.append(nn.LSTM(input_size=input_size, 
                                   hidden_size=dim_hidden, 
                                   num_layers=1, 
                                   bidirectional=bidirectional, 
                                   batch_first=True))
        # 標準のリスト型のままだと，
        # Pytorchで扱えないので，ModuleListに変換する
        self.rnn = nn.ModuleList(rnn)

        # sub_sample の定義
        if sub_sample is None:
            # 定義されていない場合は，フレームの間引きを行わない
            # (sub_sampleは全要素1のリストにする)
            self.sub_sample = [1 for i in range(num_layers)]
        else:
            # 定義されている場合は，それを用いる
            self.sub_sample = sub_sample

        # Projection層もRNN層と同様に1層ずつ定義する
        proj = []
        for n in range(self.num_layers):
            # Projection層の入力次元数 = RNN層の出力次元数．
            # bidiractional=Trueの場合は次元数が2倍になる
            input_size = dim_hidden * (2 if bidirectional else 1)
            proj.append(nn.Linear(in_features=input_size, 
                                  out_features=dim_proj))
        # RNN層と同様，ModuleListに変換する
        self.proj = nn.ModuleList(proj)
 
        
    def forward(self, sequence, lengths):
        ''' ネットワーク計算(forward処理)の関数
        sequence: 各発話の入力系列 [B x T x D]
        lengths:  各発話の系列長(フレーム数) [B]
          []の中はテンソルのサイズ
          B: ミニバッチ内の発話数(ミニバッチサイズ)
          T: 入力テンソルの系列長(ゼロ埋め部分含む)
          D: 入力次元数(dim_in)
        '''
        # 出力とその長さ情報を入力で初期化
        output = sequence
        output_lengths = lengths
 
        # num_layersの数だけ，RNNとProjection層へ交互に入力する
        for n in range(self.num_layers):
            # RNN へ入力するため，
            # 入力をPackedSequenceデータに変換する
            rnn_input \
                = nn.utils.rnn.pack_padded_sequence(output, 
                                                  output_lengths, 
                                                  batch_first=True)

            # GPUとcuDNNを使用している場合，
            # 以下の1行を入れると処理が速くなる
            # (パラメータデータのポインタをリセット)
            self.rnn[n].flatten_parameters()

            # RNN層に入力する
            output, (h, c) = self.rnn[n](rnn_input)

            # RNN出力をProjection層へ入力するため，
            # PackedSequenceデータからtensorへ戻す
            output, output_lengths \
                = nn.utils.rnn.pad_packed_sequence(output, 
                                                  batch_first=True)

            # sub sampling (間引き)の実行
            # この層における間引き率を取得
            sub = self.sub_sample[n]
            if sub > 1:
                # 間引きを実行する
                output = output[:, ::sub]
                # フレーム数を更新する 
                # 更新後のフレーム数=(更新前のフレーム数+1)//sub
                output_lengths = torch.div((output_lengths+1), sub, 
                                           rounding_mode='floor')
            # Projection層に入力する
            output = torch.tanh(self.proj[n](output))

        # sub samplingを実行した場合はフレーム数が変わるため，
        # 出力のフレーム数情報も出力する
        return output, output_lengths

