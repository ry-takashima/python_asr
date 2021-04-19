# -*- coding: utf-8 -*-

#
# ダウンロードしたラベルデータを読み込み，
# キャラクター(漢字混じりの文字)単位，かな単位，音素単位で定義されるラベルファイルを作成します．
#

# yamlデータを読み込むためのモジュールをインポート
import yaml

# osモジュールをインポート
import os

#
# メイン関数
#
if __name__ == "__main__":
    
    # ダウンロードしたラベルデータ(yaml形式)
    original_label = \
      '../data/original/jsut-label-master/text_kana/basic5000.yaml'

    # ラベルのリストを格納する場所
    out_label_dir = '../data/label/all'

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(out_label_dir, exist_ok=True)

    # ラベルデータを読み込む
    with open(original_label, mode='r') as yamlfile:
        label_info = yaml.safe_load(yamlfile)

    # キャラクター/かな/音素のラベルファイルを書き込みモードで開く
    with open(os.path.join(out_label_dir, 'text_char'), 
              mode='w') as label_char, \
              open(os.path.join(out_label_dir, 'text_kana'), 
              mode='w') as label_kana, \
              open(os.path.join(out_label_dir, 'text_phone'), 
              mode='w') as label_phone:
        # BASIC5000_0001 ~ BASIC5000_5000 に対して処理を繰り返し実行
        for i in range(5000):
            # 発話ID
            filename = 'BASIC5000_%04d' % (i+1)
            
            # 発話ID が label_info に含まれない場合はエラー
            if not filename in label_info:
                print('Error: %s is not in %s' % (filename, original_label))
                exit()

            # キャラクターラベル情報を取得
            chars = label_info[filename]['text_level2']
            # '、'と'。'を除去
            chars = chars.replace('、', '')
            chars = chars.replace('。', '')

            # かなラベル情報を取得
            kanas = label_info[filename]['kana_level3']
            # '、'を除去
            kanas = kanas.replace('、', '')

            # 音素ラベル情報を取得
            phones = label_info[filename]['phone_level3']

            # キャラクターラベルファイルへ，1文字ずつスペース区切りで書き込む
            # (' '.join(list) は，リストの各要素にスペースを挟んで，1文にする)
            label_char.write('%s %s\n' % (filename, ' '.join(chars)))

            # かなラベルファイルへ，1文字ずつスペース区切りで書き込む
            label_kana.write('%s %s\n' % (filename, ' '.join(kanas)))

            # 音素ラベルは，'-'をスペースに置換して書き込む
            label_phone.write('%s %s\n' % (filename, phones.replace('-',' ')))

