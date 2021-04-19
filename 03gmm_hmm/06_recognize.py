# -*- coding: utf-8 -*-

#
# HMMモデルで孤立単語認識を行います．
#

# hmmfunc.pyからMonoPhoneHMMクラスをインポート
from hmmfunc import MonoPhoneHMM

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# os, sysモジュールをインポート
import sys
import os

#
# メイン関数
#
if __name__ == "__main__":

    # HMMファイル
    hmm_file = './exp/model_3state_2mix/10.hmm'

    # 評価データの特徴量リストのファイル
    feat_scp = './exp/data/test/mfcc/feats.scp'

    # 辞書ファイル
    lexicon_file = './exp/data/test/lexicon.txt'

    # 音素リスト
    phone_list_file = \
        './exp/data/train_small/phone_list'

    # Trueの場合，文章の先頭と末尾に
    # ポーズがあることを前提とする
    insert_sil = True

    #
    # 処理ここから
    #

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

    # MonoPhoneHMMクラスを呼び出す
    hmm = MonoPhoneHMM()

    # HMMを読み込む
    hmm.load_hmm(hmm_file)

    # 特徴量リストファイルを開き，
    # 発話毎に音声認識を行う
    with open(feat_scp, mode='r') as f:
        for line in f:
            # 0列目は発話ID
            utt = line.split()[0]
            # 1列目はファイルパス
            ff = line.split()[1]
            # 3列目は次元数
            nd = int(line.split()[3])
            
            # 次元数がHMMの次元数と一致しなければエラー
            if hmm.num_dims != nd:
                sys.stderr.write(\
                    '%s: unexpected #dims (%d)\n'\
                    % (utt, nd))
                exit(1)

            # 特徴量ファイルを開く
            feat = np.fromfile(ff, dtype=np.float32)
            # フレーム数 x 次元数の配列に変形
            feat = feat.reshape(-1, hmm.num_dims)

            # 孤立単語認識を行う
            (result, detail) = hmm.recognize(feat, lexicon)

            # resultには最も尤度の高い単語が格納されている
            # detailは尤度のランキングが格納されている
            # 結果を出力する
            sys.stdout.write('%s %s\n' % (utt, ff))
            sys.stdout.write('Result = %s\n' % (result))
            sys.stdout.write('[Runking]\n')
            for res in detail:
                sys.stdout.write('  %s %f\n' \
                    % (res['word'], res['score']))
       
