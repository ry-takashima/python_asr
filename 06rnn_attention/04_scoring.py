# -*- coding: utf-8 -*-

#
# 認識結果と正解文を参照して，認識エラー率を計算します．
#

# 認識エラー率を計算するモジュールをインポート
import levenshtein

# os, sysモジュールをインポート
import os
import sys


if __name__ == "__main__":
 
    # トークンの単位
    # phone:音素  kana:かな  char:キャラクター
    unit = 'phone'
 
    # 実験ディレクトリ
    exp_dir = './exp_train_small'
    
    # デコード結果が格納されているディレクトリ
    decoded_dir = os.path.join(exp_dir, 
                               unit+'_model_attention',
                               'decode_test')

    # 認識結果が記述されたファイル
    hypothesis_file = os.path.join(decoded_dir, 'hypothesis.txt')

    # 正解文が記述されたファイル
    reference_file = os.path.join(decoded_dir, 'reference.txt')

    # エラー率算出結果を出力するディレクトリ
    out_dir = decoded_dir

    # エラー率算出結果の出力ファイル
    result_file = os.path.join(out_dir, 'result.txt')

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(out_dir, exist_ok=True)

    # 各誤りの総数(エラー率算出時の分子)
    total_err = 0
    total_sub = 0
    total_del = 0
    total_ins = 0
    # 正解文の総文字数(エラー率算出時の分母)
    total_length = 0

    # 各ファイルをオープン
    with open(hypothesis_file, mode='r') as hyp_file, \
         open(reference_file, mode='r') as ref_file, \
         open(result_file, mode='w') as out_file:
        # 認識結果ファイル正解文ファイルを一行ずつ読み込む
        for line_hyp, line_ref in zip(hyp_file, ref_file):
            # 読み込んだ行をスペースで区切り，リスト型の変数にする
            parts_hyp = line_hyp.split()
            parts_ref = line_ref.split()

            # 発話ID(partsの0番目の要素)が一致しているか確認
            if parts_hyp[0] != parts_ref[0]:
                sys.stderr.write('Utterance ids of '\
                    'hypothesis and reference do not match.')
                exit(1)

            # 1要素目以降が認識結果/正解分の文字列(リスト型)
            hypothesis = parts_hyp[1:]
            reference = parts_ref[1:]

            # 誤り数を計算する
            (error, substitute, delete, insert, ref_length) \
                = levenshtein.calculate_error(hypothesis,
                                              reference)

            # 総誤り数を累積する
            total_err += error
            total_sub += substitute
            total_del += delete
            total_ins += insert
            total_length += ref_length

            # 各発話の結果を出力する
            out_file.write('ID: %s\n' % (parts_hyp[0]))
            out_file.write('#ERROR (#SUB #DEL #INS): '\
                '%d (%d %d %d)\n' \
                % (error, substitute, delete, insert))
            out_file.write('REF: %s\n' % (' '.join(reference)))
            out_file.write('HYP: %s\n' % (' '.join(hypothesis)))
            out_file.write('\n')

        # 総エラー数を，正解文の総文字数で割り，エラー率を算出する
        err_rate = 100.0 * total_err / total_length
        sub_rate = 100.0 * total_sub / total_length
        del_rate = 100.0 * total_del / total_length
        ins_rate = 100.0 * total_ins / total_length
        
        # 最終結果を出力する
        out_file.write('------------------------------'\
            '-----------------------------------------------\n')
        out_file.write('#TOKEN: %d, #ERROR: %d '\
            '(#SUB: %d, #DEL: %d, #INS: %d)\n' \
            % (total_length, total_err,
               total_sub, total_del, total_ins))
        out_file.write('TER: %.2f%% (SUB: %.2f, '\
            'DEL: %.2f, INS: %.2f)\n' \
            % (err_rate, sub_rate, del_rate, ins_rate))
        print('TER: %.2f%% (SUB: %.2f, DEL: %.2f, INS: %.2f)' \
            % (err_rate, sub_rate, del_rate, ins_rate))
        out_file.write('------------------------------'\
            '-----------------------------------------------\n')

