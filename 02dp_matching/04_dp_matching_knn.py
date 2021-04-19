# -*- coding: utf-8 -*-

#
# DPマッチングにより，
# 発話内容のインデックスを推定します．
#

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# sysモジュールをインポート
import sys


def dp_matching(feature_1, feature_2):
    ''' DPマッチングを行う
    入力:
        feature_1: 比較する特徴量系列1
        feature_2: 比較する特徴量系列2
    出力:
        total_cost: 最短経路の総コスト
        min_path:   最短経路のフレーム対応
    '''
    # フレーム数と次元数を取得
    (nframes_1, num_dims) = np.shape(feature_1)
    nframes_2 = np.shape(feature_2)[0]

    # 距離(コスト)行列を計算
    distance = np.zeros((nframes_1, nframes_2))
    for n in range(nframes_1):
        for m in range(nframes_2):
            # feature_1 の n フレーム目と
            # feature_2 の m フレーム目の
            # ユークリッド距離の二乗を計算
            distance[n, m] = \
              np.sum((feature_1[n] - feature_2[m])**2)
    
    # 累積コスト行列
    cost = np.zeros((nframes_1, nframes_2))
    # 遷移の種類(縦/斜め/横)を記録する行列
    # 0: 縦の遷移, 1:斜めの遷移, 2:横の遷移
    track = np.zeros((nframes_1, nframes_2), np.int16)

    # スタート地点の距離
    cost[0, 0] = distance[0, 0]

    # 縦の縁: 必ず縦に遷移する
    for n in range(1, nframes_1):
        cost[n, 0] = cost[n-1, 0] + distance[n, 0]
        track[n, 0] = 0

    # 横の縁: 必ず横に遷移する
    for m in range(1, nframes_2):
        cost[0, m] = cost[0, m-1] + distance[0, m]
        track[0, m] = 2

    # それ以外: 縦横斜めの内，最小の遷移を行う
    for n in range(1, nframes_1):
        for m in range(1, nframes_2):
            # 縦の遷移をしたときの累積コスト
            vertical = cost[n-1, m] + distance[n, m]
            # 斜めの遷移をしたときの累積コスト
            # (斜めは2倍のペナルティを与える)
            diagonal = cost[n-1, m-1] + 2 * distance[n, m]
            # 横の遷移をしたときの累積コスト
            horizontal = cost[n, m-1] + distance[n, m]
            
            # 累積コストが最小となる遷移を選択する
            candidate = [vertical, diagonal, horizontal]
            transition = np.argmin(candidate)

            # 累積コストと遷移を記録する
            cost[n, m] = candidate[transition]
            track[n, m] = transition

    # 総コストはcost行列の最終行最終列の値
    # 特徴量のフレーム数で正規化する
    total_cost = cost[-1, -1] / (nframes_1 + nframes_2)

    #
    # バックトラック
    # 終端からtrackの値を見ながら逆に辿ることで，
    # 最小コストのパスを求める
    min_path = []
    # 終端からスタート
    n = nframes_1 - 1
    m = nframes_2 - 1
    while True:
        # 現在のフレーム番号の組をmin_pathに加える
        min_path.append([n,m])

        # スタート地点まで到達したら終了
        if n == 0 and m == 0:
            break

        # track の値を見る
        if track[n, m] == 0:
            # 縦のパスを通ってきた場合
            n -= 1
        elif track[n, m] == 1:
            # 斜めのパスを通ってきた場合
            n -= 1
            m -= 1
        else:
            # 横のパスを通ってきた場合
            m -= 1

    # min_path を逆順に並び替える
    min_path = min_path[::-1]

    # 総コストとパスを出力
    return total_cost, min_path


#
# メイン関数
#
if __name__ == "__main__":
    # 認識対象のセット番号と発話番号
    query_set = 1
    query_utt = 9

    # K-nearest neighborのパラメータ
    K = 3
  
    # MFCCの次元数
    num_dims = 13
    # 総セット数
    num_set = 5
    # 発話の種類数
    num_utt = 10
    
    # 特徴量データを特徴量ファイルから読み込む
    query_file = './mfcc/REPEAT500_set%d_%03d.bin' % \
                 (query_set, query_utt)
    query = np.fromfile(query_file, dtype=np.float32)
    query = query.reshape(-1, num_dims)

    cost = []
    for set_id in range(1, num_set+1):
        for utt_id in range(1, num_utt+1):
            # query と同じセットは比較しない
            if set_id == query_set:
                continue

            # 比較対象の特徴量を読み込む
            target_file = './mfcc/REPEAT500_set%d_%03d.bin' % \
                           (set_id, utt_id)
            print(target_file)
            target = np.fromfile(target_file, 
                                 dtype=np.float32)
            target = target.reshape(-1, num_dims)

            # DPマッチング実施
            tmp_cost, tmp_path = dp_matching(query, target)

            cost.append({'utt': utt_id,
                         'set': set_id,
                         'cost': tmp_cost
                        })

    # コストの昇順に並び替える
    cost = sorted(cost, key=lambda x:x['cost'])
    
    # コストのランキングを表示する
    for n in range(len(cost)):
        print('%d: utt: %d,  set: %d, cost: %.3f' % \
              (n+1,
               cost[n]['utt'], 
               cost[n]['set'], 
               cost[n]['cost']))

    #
    # K-nearest neighbor を行う
    #
    voting = np.zeros(num_utt, np.int16)
    for n in range(K):
        # トップK個の発話IDで投票を行う
        voting[cost[n]['utt']-1] += 1
    
    # 投票の最も大きかった発話IDを出力する
    max_voted = np.argmax(voting) + 1
    print('Estimated utterance id = %d' % max_voted)


