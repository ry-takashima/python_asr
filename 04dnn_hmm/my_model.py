# -*- coding: utf-8 -*-

#
# モデル構造を定義します
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch.nn as nn

# 作成したinitialize.pyの
# 初期化関数 lecun_initialization をインポート
from initialize import lecun_initialization


class MyDNN(nn.Module):
    ''' Fully connected layer (線形層) によるシンプルなDNN
    dim_in:     入力特徴量の次元数
    dim_hidden: 隠れ層の次元数
    dim_out:    出力の次元数
    num_layers: 隠れ層の数
    '''
    def __init__(self, 
                 dim_in,
                 dim_hidden,
                 dim_out,
                 num_layers=2):
        super(MyDNN, self).__init__()
        
        # 隠れ層の数
        self.num_layers = num_layers

        # 入力層: 線形層+ReLU
        self.inp = nn.Sequential(\
            nn.Linear(in_features=dim_in,
                      out_features=dim_hidden),
            nn.ReLU())
        # 隠れ層
        hidden = []
        for n in range(self.num_layers):
            # 線形層をhiddenに加える
            hidden.append(nn.Linear(in_features=dim_hidden,
                                    out_features=dim_hidden))
            # ReLUを加える
            hidden.append(nn.ReLU())

        # Pytorchで扱うため，リスト型から
        # ModuleList型に変換する
        self.hidden = nn.ModuleList(hidden)

        # 出力層: 線形層
        self.out = nn.Linear(in_features=dim_hidden,
                             out_features=dim_out)

        # LeCunのパラメータ初期化を実行
        lecun_initialization(self)


    def forward(self, frame):
        ''' ネットワーク計算(forward処理)の関数
        frame:  入力のフレームデータ
        output: 入力されたフレームに対する各状態の確率
        '''
        # 入力層を通す
        output = self.inp(frame)
        # 隠れ層を通す
        for n in range(self.num_layers):
            output = self.hidden[n](output)
        # 出力層を通す
        output = self.out(output)
        return output

