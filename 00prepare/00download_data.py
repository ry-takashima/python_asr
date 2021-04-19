# -*- coding: utf-8 -*-

#
# 本書で取り扱う音声データと，ラベルデータをダウンロードします．
# データは，JSUTコーパスを使用します．
# https://sites.google.com/site/shinnosuketakamichi/publication/jsut
#

# ファイルをダウンロードするためのモジュールをインポート
from urllib.request import urlretrieve

# zipファイルを展開するためのモジュールをインポート
import zipfile

# osモジュールをインポート
import os

#
# メイン関数
#
if __name__ == "__main__":
    
    # データの置き場を定義
    data_dir = '../data/original'

    # ディレクトリdata_dirが存在しない場合は作成する
    os.makedirs(data_dir, exist_ok=True)

    # 音声ファイル(jsutコーパス. zip形式)をダウンロード
    data_archive = os.path.join(data_dir, 'jsut-data.zip')
    print('download jsut-data start')
    urlretrieve('http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip', 
                data_archive)
    print('download jsut-data finished')

    # ダウンロードしたデータを展開する
    print('extract jsut-data start')
    with zipfile.ZipFile(data_archive) as data_zip:
        data_zip.extractall(data_dir)
    print('extract jsut-data finished')

    # zipファイルを削除する
    os.remove(data_archive)

    # jsutコーパスのラベルデータをダウンロード
    label_archive = os.path.join(data_dir, 'jsut-label.zip')
    print('download jsut-label start')
    urlretrieve('https://github.com/sarulab-speech/jsut-label/archive/master.zip',
                label_archive)
    print('download jsut-label finished')

    # ダウンロードしたデータを展開する
    print('extract jsut-label start')
    with zipfile.ZipFile(label_archive) as label_zip:
        label_zip.extractall(data_dir)
    print('extract jsut-label finished')

    # zipファイルを削除する
    os.remove(label_archive)

    print('all processes finished')

