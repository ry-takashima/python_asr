# -*- coding: utf-8 -*-

#
# レーベンシュタイン距離を用いて，
# 認識結果の誤り数を算出します．
#

import numpy as np
import copy

def calculate_error(hypothesis, reference):
    ''' レーベンシュタイン距離を計算し，
        置換誤り，削除誤り，挿入誤りを出力する
    hypothesis:       認識結果(トークン毎に区切ったリスト形式)
    reference:        正解(同上)
    total_error:      総誤り数
    substitute_error: 置換誤り数
    delete_error:     削除誤り数
    insert_error:     挿入誤り数
    len_ref:          正解文のトークン数
    '''
    # 認識結果および正解系列の長さを取得
    len_hyp = len(hypothesis)
    len_ref = len(reference)

    # 累積コスト行列を作成する
    # 行列の各要素には，トータルコスト，
    # 置換コスト，削除コスト，挿入コストの
    # 累積値が辞書形式で定義される．
    cost_matrix = [[{"total":0, 
                     "substitute":0,
                     "delete":0,
                     "insert":0} \
                     for j in range(len_ref+1)] \
                         for i in range(len_hyp+1)]

    # 0列目と0行目の入力
    for i in range(1, len_hyp+1):
        # 縦方向への遷移は，削除処理を意味する
        cost_matrix[i][0]["delete"] = i
        cost_matrix[i][0]["total"] = i
    for j in range(1, len_ref+1):
        # 横方向への遷移は，挿入処理を意味する
        cost_matrix[0][j]["insert"] = j
        cost_matrix[0][j]["total"] = j

    # 1列目と1行目以降の累積コストを計算していく
    for i in range(1, len_hyp+1):
        for j in range(1, len_ref+1):
            #
            # 各処理のコストを計算する
            #
            # 斜め方向の遷移時，文字が一致しない場合は，
            # 置換処理により累積コストが1増加
            substitute_cost = \
                cost_matrix[i-1][j-1]["total"] \
                + (0 if hypothesis[i-1] == reference[j-1] else 1)
            # 縦方向の遷移時は，削除処理により累積コストが1増加
            delete_cost = cost_matrix[i-1][j]["total"] + 1
            # 横方向の遷移時は，挿入処理により累積コストが1増加
            insert_cost = cost_matrix[i][j-1]["total"] + 1

            # 置換処理，削除処理，挿入処理のうち，
            # どの処理を行えば累積コストが最も小さくなるかを計算
            cost = [substitute_cost, delete_cost, insert_cost]
            min_index = np.argmin(cost)

            if min_index == 0:
                # 置換処理が累積コスト最小となる場合

                # 遷移元の累積コスト情報をコピー
                cost_matrix[i][j] = \
                    copy.copy(cost_matrix[i-1][j-1])
                # 文字が一致しない場合は，
                # 累積置換コストを1増加させる
                cost_matrix[i][j]["substitute"] \
                    += (0 if hypothesis[i-1] \
                        == reference[j-1] else 1)
            elif min_index == 1:
                # 削除処理が累積コスト最小となる場合
                
                # 遷移元の累積コスト情報をコピー
                cost_matrix[i][j] = copy.copy(cost_matrix[i-1][j])
                # 累積削除コストを1増加させる
                cost_matrix[i][j]["delete"] += 1
            else:
                # 置換処理が累積コスト最小となる場合
                
                # 遷移元の累積コスト情報をコピー
                cost_matrix[i][j] = copy.copy(cost_matrix[i][j-1])
                # 累積挿入コストを1増加させる
                cost_matrix[i][j]["insert"] += 1

            # 累積トータルコスト(置換+削除+挿入コスト)を更新
            cost_matrix[i][j]["total"] = cost[min_index]

    #
    # エラーの数を出力する
    # このとき，削除コストは挿入誤り，
    # 挿入コストは削除誤りになる点に注意．
    # (削除コストが1である
    #    = 1文字削除しないと正解文にならない 
    #    = 認識結果は1文字分余計に挿入されている
    #    = 挿入誤りが1である)
    #

    # 累積コスト行列の右下の要素が最終的なコストとなる．
    total_error = cost_matrix[len_hyp][len_ref]["total"]
    substitute_error = cost_matrix[len_hyp][len_ref]["substitute"]
    # 削除誤り = 挿入コスト
    delete_error = cost_matrix[len_hyp][len_ref]["insert"]
    # 挿入誤り = 削除コスト
    insert_error = cost_matrix[len_hyp][len_ref]["delete"]
    
    # 各誤り数と，正解文の文字数
    # (誤り率を算出する際に分母として用いる)を出力
    return (total_error, 
            substitute_error,
            delete_error,
            insert_error,
            len_ref)


if __name__ == "__main__":
    # ref: 正解文
    # hyp: 認識結果
    ref = "狼が犬に似ているようにおべっか使いは友達のように見える"
    hyp = "オオ狼みがい塗に似ているようにオッつ界は伴ちのように見える"


    # 各文字列を，1文字ずつ区切ってリストデータにする
    hyp_list = list(hyp)
    ref_list = list(ref)

    # 誤り数を計算する
    total, substitute, delete, insert, ref_length \
        = calculate_error(hyp_list, ref_list)

    # 誤りの数と，誤り率(100*誤り数/正解文の文字数)を出力する
    print("REF: %s" % ref)
    print("HYP: %s" % hyp)
    print("#TOKEN(REF): %d, #ERROR: %d, #SUB: %d, #DEL: %d, #INS: %d" \
        % (ref_length, total, substitute, delete, insert))
    print("UER: %.2f, SUBR: %.2f, DELR: %.2f, INSR: %.2f" \
        % (100.0*total/ref_length,
           100.0*substitute/ref_length,
           100.0*delete/ref_length,
           100.0*insert/ref_length))

