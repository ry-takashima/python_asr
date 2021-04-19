# -*- coding: utf-8 -*-

#
# モデルの初期化関数を定義します
#

# 数値演算用モジュール(numpy)をインポート
import numpy as np


def lecun_initialization(model):
    '''LeCunのパラメータ初期化方法の実行
    各重み(バイアス成分除く)を，平均0，標準偏差 1/sqrt(dim) の
    正規分布に基づく乱数で初期化(dim は入力次元数)
    model: Pytorchで定義したモデル
    '''
    # モデルのパラメータを順に取り出し，初期化を実行する
    for param in model.parameters():
        # パラメータの値を取り出す
        data = param.data
        # パラメータのテンソル次元数を取り出す
        dim = data.dim()
        # 次元数を元に処理を変える
        if dim == 1:
            # dim = 1 の場合はバイアス成分
            # ゼロに初期化する
            data.zero_()
        elif dim == 2:
            # dim = 2 の場合は線形射影の重み行列
            # 入力次元数 = size(1) を得る
            n = data.size(1)
            # 入力次元数の平方根の逆数を
            # 標準偏差とする正規分布乱数で初期化
            std = 1.0 / np.sqrt(n)
            data.normal_(0, std)
        elif dim == 3:
            # dim = 3 の場合は 1次元畳み込みの行列
            # 入力チャネル数 * カーネルサイズの
            # 平方根の逆数を標準偏差とする
            # 正規分布乱数で初期化
            n = data.size(1) * data.size(2)
            std = 1.0 / np.sqrt(n)
            data.normal_(0, std)
        elif dim == 4:
            # dim = 4 の場合は 2次元畳み込みの行列
            # 入力チャネル数 * カーネルサイズ(行) 
            #   * カーネルサイズ(列) 
            # の平方根の逆数を標準偏差とする
            # 正規分布乱数で初期化
            n = data.size(1) * data.size(2) * data.size(3)
            std = 1.0 / np.sqrt(n)
            data.normal_(0, std)
        else:
            # それ以外は対応していない
            print('lecun_initialization: '\
                  'dim > 4 is not supported.')
            exit(1)

