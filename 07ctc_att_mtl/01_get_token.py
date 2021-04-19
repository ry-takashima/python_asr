# -*- coding: utf-8 -*-

#
# ラベル中のトークン(音素/かな/キャラクター)を，
# ニューラルネットワークで扱うため，文字から番号へ変換します．
# また，文字－番号の対応を記したトークンリストも作成します．
#

# osモジュールをインポート
import os


def token_to_int(label_file_str,
                 label_file_int,
                 unknown_list_file,
                 token_list,
                 silence_tokens):
    ''' トークンリストを使って，ラベルファイルの
        文字を番号に置き換えてファイルに出力する
        このとき，非音声トークンは削除する
    label_file_str:    文字で記述されたラベルファイル
    label_file_int:    番号で記述されたラベルファイルの出力先
    unknown_list_file: 未知のトークンを記述したファイルの出力先
    token_list:        文字－番号の対応情報が格納されたリスト
    silence_tokenas:   非音声トークンが格納されたリスト
    '''
    # 各ファイルを開く
    with open(label_file_str, mode='r') as label_in, \
             open(label_file_int, mode='w') as label_out, \
             open(unknown_list_file, mode='w') as unk_list:
        # ラベルファイルを一行ずつ読み込む
        for line in label_in:
            # 読み込んだ行をスペースで区切り，
            # リスト型の変数にする
            text = line.split()
            
            # リストの0番目の要素は発話IDなので，そのまま出力する
            label_out.write('%s' % text[0])

            # リストの1番目以降の要素は文字なので，
            # 1文字ずつ数字に置き換える
            for u in text[1:]:
                # 非音声トークンの場合はスキップする
                if u in silence_tokens:
                    continue

                # 文字がトークンリストに存在するか確認
                if not u in token_list:
                    # 存在しなかった場合
                    # 未知トークンリストに書き込む
                    unk_list.write('%s\n' % (u))
                    # 未知トークンには番号 1 を割り当てて出力
                    label_out.write(' 1')
                else:
                    # 存在する場合
                    # 対応する番号を出力
                    label_out.write(' %d' \
                                    % (token_list.index(u) + 1))
            label_out.write('\n')


#
# メイン関数
#
if __name__ == "__main__":
    
    # トークンの単位
    # phone:音素  kana:かな  char:キャラクター
    unit_set = ['phone', 'kana', 'char']

    # 非音声トークン(ポーズなど)の定義
    # 本プログラムでは，非音声トークンは扱わない
    # 0行目は音素，1行目はかな，2行目は文字の
    # 非音声トークンを定義する
    silence_set = [['pau'],
                   [''],
                   ['']]

    # 未知トークンの定義
    # 訓練データに無く，開発/評価データに有った
    # トークンはこの文字に置き換えられる
    unknown_token = '<unk>'

    # ラベルファイルの存在するフォルダ
    label_dir_train = '../data/label/train_small'
    label_dir_dev = '../data/label/dev'
    label_dir_test = '../data/label/test'

    # 実験ディレクトリ
    # train_smallを使った時とtrain_largeを使った時で
    # 異なる実験ディレクトリにする
    exp_dir = './exp_' + os.path.basename(label_dir_train) 

    # 処理結果の出力先ディレクトリ
    output_dir = os.path.join(exp_dir, 'data')
    
    # phone, kana, char それぞれに対して処理
    for uid, unit in enumerate(unit_set):
        # 出力先ディレクトリ
        out_dir = os.path.join(output_dir, unit)

        # 出力ディレクトリが存在しない場合は作成する
        os.makedirs(out_dir, exist_ok=True)

        #
        # トークンリストを作成
        #
        # 学習データのラベルファイル
        label_train = os.path.join(label_dir_train, 'text_'+unit)
        # トークンリストを空リストで定義
        token_list = []
        # 訓練データのラベルに存在するトークンを
        # token_listに登録する
        with open(label_train) as label_file:
            # 一行ずつ読み込む
            for line in label_file:
                # 読み込んだ行をスペースで区切り，
                # リスト型の変数にする
                # 0番目は発話IDなので，1番目以降を取り出す
                text = line.split()[1:]
                
                # トークンリストに結合
                token_list += text

                # set関数により，重複するトークンを削除する
                token_list = list(set(token_list))

        # 非音声トークン
        silence_tokens = silence_set[uid]
        # 非音声トークンをリストから削除する
        for u in silence_tokens:
            if u in token_list:
                token_list.remove(u)

        # リストをソートする
        token_list = sorted(token_list)

        # 未知トークンをリストの先頭に挿入する
        token_list.insert(0, unknown_token)

        # トークンリストをファイルに出力する
        with open(os.path.join(out_dir, 'token_list'), 
                  mode='w') as token_file:
            for i, u in enumerate(token_list):
                # 「トークン 対応する番号」を記述
                # このとき，番号は1始まりにする．
                # (0はCTCのblankトークンに割り当てるため)
                token_file.write('%s %d\n' % (u, i+1))

        #
        # 作成したトークンリストを使って，ラベルファイルの
        # 各トークンを文字から番号に置き換える
        #
        # 開発/評価データに存在するトークンで，token_listに
        # (=学習データに)存在しないトークンは未知トークンとして
        # unknown_listに登録する．
        # (unknown_listは以降の処理には使いませんが，
        # 処理結果を確認するために出力しています)
        #
        # 学習/開発/評価データのラベルそれぞれについて処理
        label_dir_list = [label_dir_train,
                          label_dir_dev,
                          label_dir_test]
        for label_dir in label_dir_list:
            # ディレクトリ名を取得(train_{small,large}/dev/test)
            name = os.path.basename(label_dir)
            # 入力ラベルファイル(各トークンが文字で表記)
            label_str = os.path.join(label_dir,
                                     'text_'+unit)
            # 出力ラベルファイル(各トークンが数値で表記)
            label_int = os.path.join(out_dir,
                                     'label_'+name)
            # 未知トークンリストの出力先
            unknown_list = os.path.join(out_dir, 
                                        'unknown_token_'+name)
            # ラベルファイルの文字->数値変換処理の実行
            token_to_int(label_str,
                         label_int,
                         unknown_list,
                         token_list,
                         silence_tokens)

