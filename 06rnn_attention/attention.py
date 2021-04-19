# -*- coding: utf-8 -*-

#
# Attention (Location aware attention) の実装です．
# 参考文献
#   - D. Bahdanau, et al., 
#     ``End-to-end attention-based large vocabulary speech
#       recognition,''
#     in Proc. ICASSP, 2016.
#   - J. Chorowski, et al.,
#     ``Attention-based models for speech recognition,''
#     in Proc. NIPS , 2015.
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocationAwareAttention(nn.Module):
    ''' Location aware attention
    dim_encoder:   エンコーダRNN出力の次元数
    dim_decoder:   デコーダRNN出力の次元数
    dim_attention: Attention機構の次元数
    filter_size:   location filter (前のAttention重みに
                   畳み込まれるフィルタ)のサイズ
    filter_num:    location filterの数
    temperature:   Attention重み計算時に用いる温度パラメータ
    '''
    def __init__(self,
                 dim_encoder,
                 dim_decoder,
                 dim_attention,
                 filter_size, 
                 filter_num,
                 temperature=1.0):

        super(LocationAwareAttention, self).__init__()

        # F: 前のAttention重みに畳み込まれる畳み込み層
        self.loc_conv = nn.Conv1d(in_channels=1,
                                  out_channels=filter_num, 
                                  kernel_size=2*filter_size+1,
                                  stride=1, 
                                  padding=filter_size,
                                  bias=False)
        # 以下三つの層のうち，一つのみbiasをTrueにし，他はFalseにする
        # W: 前ステップのデコーダRNN出力にかかる射影層
        self.dec_proj = nn.Linear(in_features=dim_decoder, 
                                  out_features=dim_attention,
                                  bias=False)
        # V: エンコーダRNN出力にかかる射影層
        self.enc_proj = nn.Linear(in_features=dim_encoder, 
                                  out_features=dim_attention,
                                  bias=False)
        # U: 畳み込み後のAttention重みにかかる射影層
        self.att_proj = nn.Linear(in_features=filter_num, 
                                  out_features=dim_attention,
                                  bias=True)
        # w: Ws + Vh + Uf + b にかかる線形層
        self.out = nn.Linear(in_features=dim_attention,
                             out_features=1)

        # 各次元数
        self.dim_encoder = dim_encoder
        self.dim_decoder = dim_decoder
        self.dim_attention = dim_attention

        # 温度パラメータ
        self.temperature = temperature

        # エンコーダRNN出力(h)とその射影(Vh)
        # これらは毎デコードステップで同じ値のため，
        # 一回のみ計算し，計算結果を保持しておく
        self.input_enc = None
        self.projected_enc = None
        # エンコーダRNN出力の，発話ごとの系列長
        self.enc_lengths = None
        # エンコーダRNN出力の最大系列長
        # (=ゼロ詰めしたエンコーダRNN出力の系列長)
        self.max_enc_length = None
        # Attentionマスク
        # エンコーダの系列長以降
        # (ゼロ詰めされている部分)の重みをゼロにするマスク
        self.mask = None


    def reset(self):
        ''' 内部パラメータのリセット
            この関数は1バッチの処理を行うたびに，
            最初に呼び出す必要がある
        '''
        self.input_enc = None
        self.projected_enc = None
        self.enc_lengths = None
        self.max_enc_length = None
        self.mask = None

    
    def forward(self, 
                input_enc,
                enc_lengths,
                input_dec=None,
                prev_att=None):
        ''' ネットワーク計算(forward処理)の関数
        input_enc:   エンコーダRNNの出力 [B x Tenc x Denc]
        enc_lengths: バッチ内の各発話のエンコーダRNN出力の系列長 [B]
        input_dec:   前ステップにおけるデコーダRNNの出力 [B x Ddec]
        prev_att:    前ステップにおけるAttention重み [B x Tenc]
          []の中はテンソルのサイズ
          B:    ミニバッチ内の発話数(ミニバッチサイズ)
          Tenc: エンコーダRNN出力の系列長(ゼロ埋め部分含む)
          Denc: エンコーダRNN出力の次元数(dim_encoder)
          Ddec: デコーダRNN出力の次元数(dim_decoder)
        '''
        # バッチサイズ(発話数)を得る
        batch_size = input_enc.size()[0]

        #
        # エンコーダRNN出力とその射影ベクトルを一度のみ計算
        #
        if self.input_enc is None:
            # エンコーダRNN出力(h)
            self.input_enc = input_enc
            # 各発話の系列長
            self.enc_lengths = enc_lengths
            # 最大系列長
            self.max_enc_length = input_enc.size()[1]
            # 射影を行う(Vhの計算)
            self.projected_enc = self.enc_proj(self.input_enc)
        
        #
        # 前ステップにおけるデコーダRNN出力を射影する(Wsの計算)
        #
        # 前のデコーダRNN出力が無い場合は初期値としてゼロ行列を使用
        if input_dec is None:
            input_dec = torch.zeros(batch_size, self.dim_decoder)
            # 作成したテンソルをエンコーダRNN出力と
            # 同じデバイス(GPU/CPU)に配置
            input_dec = input_dec.to(device=self.input_enc.device, 
                                     dtype=self.input_enc.dtype)
        # 前のデコーダRNN出力を射影する
        projected_dec = self.dec_proj(input_dec)

        #
        # 前ステップにおけるのAttention重み情報を
        # 射影する(Uf+bの計算)
        #
        # Attentionマスクを作成
        if self.mask is None:
            self.mask = torch.zeros(batch_size, 
                                    self.max_enc_length, 
                                    dtype=torch.bool)
            # バッチ内の各発話について，その発話の
            # 系列長以上の要素(つまりゼロ埋めされている部分)を
            # 1(=マスキング対象)にする
            for i, length in enumerate(self.enc_lengths):
                length = length.item()
                self.mask[i, length:] = 1
            # 作成したテンソルをエンコーダRNN出力と
            # 同じデバイス(GPU/CPU)に配置
            self.mask = self.mask.to(device=self.input_enc.device)

        # 前のAttention重みが無い場合は初期値として，
        # 一様の重みを与える
        if prev_att is None:
            # 全ての要素を1のテンソルを作成
            prev_att = torch.ones(batch_size, self.max_enc_length)
            # 発話毎の系列長で割る
            # このとき，prev_attは2次のテンソル，
            # enc_lengthsは1次のテンソルなので，
            # view(batch_size, 1)によりenc_lengthsを
            # 2次テンソルの形にしてから割り算する
            prev_att = prev_att \
                     / self.enc_lengths.view(batch_size, 1)
            # 作成したテンソルをエンコーダRNN出力と
            # 同じデバイス(GPU/CPU)に配置
            prev_att = prev_att.to(device=self.input_enc.device, 
                                   dtype=self.input_enc.dtype)
            # 発話長以降の重みをゼロにするようマスキングを実行
            prev_att.masked_fill_(self.mask, 0)

        # Attention重みの畳み込みを計算する {f} = F*a
        # このとき，Conv1Dが受け付ける入力のサイズは
        # (batch_size, in_channels, self.max_enc_length)
        # (in_channelsは入力のチャネル数で，
        # このプログラムではin_channels=1) 
        # サイズを合わせるため，viewを行う
        convolved_att \
            = self.loc_conv(prev_att.view(batch_size, 
                                          1, self.max_enc_length))
 
        # convolved_attのサイズは
        # (batch_size, filter_num, self.max_enc_length)
        # Linearレイヤーが受け付ける入力のサイズは
        # (batch_size, self.max_enc_length, filter_num) なので，
        # transposeにより1次元目と2次元目をの入れ替えた上で
        # att_projに通す
        projected_att = self.att_proj(convolved_att.transpose(1, 2))
        
        #
        # Attention重みを計算する
        # 
        # この時点での各テンソルのサイズは
        # self.projected_enc: (batch_size, self.max_enc_length, 
        #                      self.dim_attention)
        # projected_dec: (batch_size, self.dim_attention)
        # projected_att: (batch_size, self.max_enc_length, self.dim_attention)
        # projected_decのテンソルの次元数を合わせるため，viewを用いる
        projected_dec = projected_dec.view(batch_size,
                                           1,
                                           self.dim_attention)

        # scoreを計算するため，各射影テンソルの加算，
        # tanh，さらに射影を実施
        # w tanh(Ws + Vh + Uf + b)
        score = self.out(torch.tanh(projected_dec \
                                    + self.projected_enc 
                                    + projected_att))

        # 現時点のscoreのテンソルサイズは
        # (batch_size, self.max_enc_length, 1)
        # viewを用いて元々のattentionのサイズに戻す
        score = score.view(batch_size, self.max_enc_length)

        # マスキングを行う
        # (エンコーダRNN出力でゼロ埋めされている部分の
        # 重みをゼロにする)
        # ただし，この後softmax関数の中で計算される
        # exp(score)がゼロになるように
        # しないといけないので，scoreの段階では0ではなく，
        # 0の対数値である-infで埋めておく
        score.masked_fill_(self.mask, -float('inf'))

        # 温度付きSoftmaxを計算することで，Attention重みを得る
        att_weight = F.softmax(self.temperature * score, dim=1)

        # att_weightを使って，エンコーダRNN出力の重みづけ和を計算し，
        # contextベクトルを得る
        # (viewによりinput_encとattention_weightの
        # テンソルサイズを合わせている)
        context \
            = torch.sum(self.input_enc * \
                att_weight.view(batch_size, self.max_enc_length, 1),
                dim=1)

        # contextベクトルとattention重みを出力
        return context, att_weight

