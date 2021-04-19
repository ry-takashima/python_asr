# -*- coding: utf-8 -*-

#
# DPマッチング用のデータを作成します．
# このプログラムは00_prepare/01prepare_wav.py
# を流用しています．
#

# サンプリング周波数を変換するためのモジュール(sox)をインポート
import sox

# osモジュールをインポート
import os

#
# メイン関数
#
if __name__ == "__main__":
    
    # wavファイルが展開されたディレクトリ
    # データは repeat500 を使用する
    original_wav_dir = '../data/original/jsut_ver1.1/repeat500/wav'

    # フォーマット変換したwavファイルを出力するディレクトリ
    out_wav_dir = './wav'

    # repeat500内で使用するセット数
    num_set = 5

    # repeat500内で使用する1セットあたりの発話数
    num_utt_per_set = 10

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(out_wav_dir, exist_ok=True)

    # soxによる音声変換クラスを呼び出す
    tfm = sox.Transformer()
    # サンプリング周波数を 16000Hz に変換するよう設定する
    tfm.convert(samplerate=16000)

    # セット x 発話数分だけ処理を実行
    for set_id in range(num_set):
        for utt_id in range(num_utt_per_set):
            # wavファイル名
            filename = 'REPEAT500_set%d_%03d' % (set_id+1, utt_id+1)
            # 変換元のオリジナルデータ (48000Hz)のファイル名
            wav_path_in = os.path.join(original_wav_dir, filename+'.wav')
            # 変換後のデータ(16000Hz)の保存ファイル名
            wav_path_out = os.path.join(out_wav_dir, filename+'.wav')

            print(wav_path_in)
            # ファイルが存在しない場合はエラー
            if not os.path.exists(wav_path_in):
                print('Error: Not found %s' % (wav_path_in))
                exit()

            # サンプリング周波数の変換と保存を実行する
            tfm.build_file(input_filepath=wav_path_in, 
                           output_filepath=wav_path_out)
