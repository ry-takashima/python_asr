# -*- coding: utf-8 -*-

#
# データのリストを，学習/開発/評価用のデータセットに分割します．
# ここでは，以下のように分割します．
# BASIC5000_0001~0250 : 評価データ
# BASIC5000_0251~0500 : 開発データ
# BASIC5000_0501~1500 : 学習データ（小）
# BASIC5000_0501~5000 : 学習データ（大）
#

# osモジュールをインポート
import os

#
# メイン関数
#
if __name__ == "__main__":
    
    # 全データが記述されているリストの置き場
    all_dir = '../data/label/all'

    # 評価データが記述されたリストの出力先
    out_eval_dir = '../data/label/test'
    # 開発データが記述されたリストの出力先
    out_dev_dir = '../data/label/dev'
    # 学習データ（小）が記述されたリストの出力先
    out_train_small_dir = '../data/label/train_small'
    # 学習データ（大）が記述されたリストの出力先
    out_train_large_dir = '../data/label/train_large'

    # 各出力ディレクトリが存在しない場合は作成する
    for out_dir in [out_eval_dir, out_dev_dir, 
                    out_train_small_dir, out_train_large_dir]:
        os.makedirs(out_dir, exist_ok=True)
    
    # wav.scp, text_char, text_kana, text_phoneそれぞれに同じ処理を行う
    for filename in ['wav.scp', 'text_char', 
                     'text_kana', 'text_phone']:
        # 全データを読み込みモードで，/評価/開発/学習データリストを書き込みモードで開く
        with open(os.path.join(all_dir, filename), 
                  mode='r') as all_file, \
                  open(os.path.join(out_eval_dir, filename), 
                  mode='w') as eval_file, \
                  open(os.path.join(out_dev_dir, filename), 
                  mode='w') as dev_file, \
                  open(os.path.join(out_train_small_dir, filename), 
                  mode='w') as train_small_file, \
                  open(os.path.join(out_train_large_dir, filename), 
                  mode='w') as train_large_file:
            # 1行ずつ読み込み，評価/開発/学習データリストに書き込んでいく
            for i, line in enumerate(all_file):
                if i < 250:
                    # 1~250: 評価データリストへ書き込み
                    eval_file.write(line)
                elif i < 500:
                    # 251~500: 開発データリストへ書き込み
                    dev_file.write(line)
                else:
                    # 501~5000: 学習（大）データリストへ書き込み
                    train_large_file.write(line)
                    if i < 1500:
                        # 501～1500: 学習（小）データリストへ書き込み
                        train_small_file.write(line)

