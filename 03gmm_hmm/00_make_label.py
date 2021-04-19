# -*- coding: utf-8 -*-

#
# ラベルファイルの音素をIDに変換して保存します．
# また、音素とIDの対応を記したリストを出力します．
#

# osモジュールをインポート
import os

def phone_to_int(label_str, 
                 label_int, 
                 phone_list,
                 insert_sil=False):
    ''' 
    音素リストを使ってラベルファイルの
    音素を数値に変換する
    label_str:  文字で記述されたラベルファイル
    labelint:   文字を数値に変換した書き込み先の
                ラベルファイル
    phone_list: 音素リスト
    insert_sil: Trueの場合，テキストの最初と最後に
                空白を挿入する
    '''
    # 各ファイルを開く
    with open(label_str, mode='r') as f_in, \
            open(label_int, mode='w') as f_out:
        # ラベルファイルを一行ずつ読み込む
        for line in f_in:
            # 読み込んだ行をスペースで区切り，
            # リスト型の変数にする
            text = line.split()
            
            # リストの0番目の要素は発話IDなので，
            # そのまま出力する
            f_out.write('%s' % text[0])

            # insert_silがTrueなら，
            # 先頭に0(ポーズ)を挿入
            if insert_sil:
                f_out.write(' 0')

            # リストの1番目以降の要素は文字なので，
            # 1文字ずつ数字に置き換える
            for u in text[1:]:
                # 音素リストに無い場合はエラー
                if not u in phone_list:
                    sys.stderr.write('phone_to_int: \
                        unknown phone %s\n' % u)
                    exit(1)
                # 音素のインデクスを出力
                f_out.write(' %d' % \
                    (phone_list.index(u)))

            # insert_silがTrueなら，
            # 末尾に0(ポーズ)を挿入
            if insert_sil:
                f_out.write(' 0')
            # 改行
            f_out.write('\n')


#
# メイン関数
#
if __name__ == "__main__":
    # 訓練データのラベルファイルのパス
    label_train_str = \
        '../data/label/train_small/text_phone'

    # 訓練データの処理結果の出力先ディレクトリ
    out_train_dir = \
        './exp/data/train_small'

    # 開発データのラベルファイルのパス
    # (開発データはGMM-HMMには使いませんが，
    # DNN-HMMで使用します．)
    label_dev_str = \
        '../data/label/dev/text_phone'

    # 開発データの処理結果の出力先ディレクトリ
    out_dev_dir = \
        './exp/data/dev'

    # 音素リスト
    phone_file = './phones.txt'

    # ポーズを表す記号
    silence_phone = 'pau'

    # Trueの場合，文章の先頭と末尾にポーズを挿入する．
    insert_sil = True

    # 音素リストの先頭にはポーズ記号を入れておく
    phone_list = [silence_phone]
    # 音素リストファイルを開き，phone_listに格納
    with open(phone_file, mode='r') as f:
        for line in f:
            # 空白や改行を消して音素記号を取得
            phone = line.strip()
            # 音素リストの末尾に加える
            phone_list.append(phone)


    # 訓練/開発データの情報をリスト化
    label_str_list = [label_train_str,
                      label_dev_str]
    out_dir_list = [out_train_dir,
                    out_dev_dir]

    # 訓練/開発データについてそれぞれ処理
    for (label_str, out_dir) \
           in zip(label_str_list, out_dir_list):

        # 出力ディレクトリが存在しない場合は作成する
        os.makedirs(out_dir, exist_ok=True)

        # 音素と数値の対応リストを出力
        out_phone_list = \
            os.path.join(out_dir, 'phone_list')
        with open(out_phone_list, 'w') as f:
            for i, phone in enumerate(phone_list):
                # リストに登録されている順番を
                # その音素に対応する数値とする
                f.write('%s %d\n' % (phone, i))
     
        # ラベルの音素記号を数字に変換して出力
        label_int = \
            os.path.join(out_dir, 'text_int')
        phone_to_int(label_str, 
                     label_int, 
                     phone_list, 
                     insert_sil)

