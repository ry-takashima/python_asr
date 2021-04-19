# -*- coding: utf-8 -*-

#
# HMMテストデータを作成します．
# COUNTERSUFFIX26_01.wav を単語毎に区切り，
# 16kHzのwavデータを作成します．
#

# サンプリング周波数を変換するためのモジュール(sox)をインポート
import sox

# wavデータを読み込むためのモジュール(wave)をインポート
import wave

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# osモジュールをインポート
import os

#
# メイン関数
#
if __name__ == "__main__":
    
    # データは COUNTERSUFFIX26_01 を使用する
    original_wav = \
      '../data/original/jsut_ver1.1/'\
      'countersuffix26/wav/COUNTERSUFFIX26_01.wav'
    
    # 単語情報
    word_info = [ {'word': '一つ',
                   'phones': 'h i t o ts u',
                   'time': [0.17, 0.90]},
                  {'word': '二つ',
                   'phones': 'f u t a ts u',
                   'time': [1.23, 2.02]},
                  {'word': '三つ',
                   'phones': 'm i cl ts u',
                   'time': [2.38, 3.11]},
                  {'word': '四つ',
                   'phones': 'y o cl ts u',
                   'time': [3.42, 4.10]},
                  {'word': '五つ',
                   'phones': 'i ts u ts u',
                   'time': [4.45, 5.13]},
                  {'word': '六つ',
                   'phones': 'm u cl ts u',
                   'time': [5.52, 6.15]},
                  {'word': '七つ',
                   'phones': 'n a n a ts u',
                   'time': [6.48, 7.15]},
                  {'word': '八つ',
                   'phones': 'y a cl ts u',
                   'time': [7.52, 8.17]},
                  {'word': '九つ',
                   'phones': 'k o k o n o ts u',
                   'time': [8.51, 9.31]},
                  {'word': 'とお',
                   'phones': 't o o',
                   'time': [9.55, 10.10]}
                ]

    # 音素リスト
    phone_list_file = \
        './exp/data/train_small/phone_list'

    # 結果出力ディレクトリ
    out_dir = './exp/data/test'

    # 加工した波形の出力ディレクトリ
    out_wav_dir = os.path.join(out_dir, 'wav')

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(out_wav_dir, exist_ok=True)

    # soxによる音声変換クラスを呼び出す
    tfm = sox.Transformer()
    # サンプリング周波数を 16000Hz に変換するよう設定する
    tfm.convert(samplerate=16000)

    downsampled_wav = os.path.join(out_wav_dir, 
                                   os.path.basename(original_wav))

    # ファイルが存在しない場合はエラー
    if not os.path.exists(original_wav):
        print('Error: Not found %s' % (original_wav))
        exit()

    # サンプリング周波数の変換と保存を実行する
    tfm.build_file(input_filepath=original_wav, 
                   output_filepath=downsampled_wav)

    # ダウンサンプリングした音声を読み込む
    with wave.open(downsampled_wav) as wav:
        # サンプリング周波数
        sample_frequency = wav.getframerate()
        # wavデータのサンプル数
        num_samples = wav.getnframes()
        # wavデータを読み込む
        waveform = wav.readframes(num_samples)
        # 数値(整数)に変換する
        waveform = np.frombuffer(waveform, dtype=np.int16)
        
    # wavファイルのリストファイル
    wav_scp = os.path.join(out_dir, 'wav.scp')
    with open(wav_scp, mode='w') as scp_file:
        # 各単語の波形を切り出して保存する   
        for n, info in enumerate(word_info):
            # 単語の時間情報を得る
            time = np.array(info['time'])
            # 時刻[秒]をサンプル点に変換
            time = np.int64(time * sample_frequency)
            # 単語の区間を切り出す
            cut_wav = waveform[time[0] : time[1]].copy()

            # 切り出した波形の保存ファイル名
            out_wav = os.path.join(out_wav_dir,
                                   "%d.wav" % (n+1))
            # 切り出した波形を保存する
            with wave.open(out_wav, 'w') as wav:
                # チャネル数，サンプルサイズ，
                # サンプリング周波数を設定
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_frequency)
                # 波形の書き出し
                wav.writeframes(cut_wav)

            # wavファイルのリストに書き込む
            scp_file.write('%d %s\n' % 
                           ((n+1), os.path.abspath(out_wav)))
 

    # 各単語と音素の組み合わせリスト(辞書)を作成する
    lexicon = os.path.join(out_dir, 'lexicon.txt')
    with open(lexicon, mode='w') as f:
        for info in word_info:
            f.write('%s %s\n' \
                % (info['word'], info['phones']))

    #
    # 以下は正解ラベルを作成
    # (音素アライメントのテストで使用)
    #

    # 音素リストファイルを開き，phone_listに格納
    phone_list = []
    with open(phone_list_file, mode='r') as f:
        for line in f:
            # 音素リストファイルから音素を取得
            phone = line.split()[0]
            # 音素リストの末尾に加える
            phone_list.append(phone)

    # 正解ラベルリスト(音素は数値表記)を作成
    label_file = os.path.join(out_dir, 'text_int')
    with open(label_file, mode='w') as f:
        for n, info in enumerate(word_info):
            label = info['phones'].split()
            # 両端にポーズを追加
            label.insert(0, phone_list[0])
            label.append(phone_list[0])
            # phone_listを使って音素を数値に変換し，書き込む
            f.write('%d' % (n+1))
            for ph in label:
                f.write(' %d' % (phone_list.index(ph)))
            f.write('\n')

