# -*- coding: utf-8 -*-

#
# ダウンロードしたwavファイルを，サンプリングレート16000Hzのデータに変換します．
# また，変換したwavデータのリストを作成します．
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
    original_wav_dir = '../data/original/jsut_ver1.1/basic5000/wav'

    # フォーマット変換したwavファイルを出力するディレクトリ
    out_wav_dir = '../data/wav'

    # wavデータのリストを格納するディレクトリ
    out_scp_dir = '../data/label/all'

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(out_wav_dir, exist_ok=True)
    os.makedirs(out_scp_dir, exist_ok=True)

    # soxによる音声変換クラスを呼び出す
    tfm = sox.Transformer()
    # サンプリング周波数を 16000Hz に変換するよう設定する
    tfm.convert(samplerate=16000)

    # wavデータのリストファイルを書き込みモードで開き，以降の処理を実施する
    with open(os.path.join(out_scp_dir, 'wav.scp'), mode='w') as scp_file:
        # BASIC5000_0001.wav ~ BASIC5000_5000.wav に対して処理を繰り返し実行
        for i in range(5000):
            filename = 'BASIC5000_%04d' % (i+1)
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

            # wavファイルのリストを書き込む
            scp_file.write('%s %s\n' % 
                           (filename, os.path.abspath(wav_path_out)))
        
