# -*- coding: utf-8 -*-

#
# HMMクラス
#

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# json形式の入出力を行うモジュールをインポート
import json

import sys

class MonoPhoneHMM():
    ''' HMMクラス
    Monophone HMMを定義
    Left-to-right 型
    共分散行列は対角行列を仮定
    '''
    def __init__(self):
        # 音素リスト
        self.phones = []
        # 音素数
        self.num_phones = 1
        # 各音素HMMの状態数
        self.num_states = 1
        # GMMの混合数
        self.num_mixture = 1
        # 特徴量ベクトルの次元数
        self.num_dims = 1
        # 正規分布(Single Gaussian Model: SGM)
        # のパラメータ
        self.pdf = None
        # 遷移確率(対数値)
        self.trans = None
        # log(0)の近似値
        self.LZERO = -1E10
        # 確率の計算に加える値の最小値
        # 計算量削減のため，この値より小さい確率は
        # 一部の計算において無視される
        self.LSMALL = -0.5E10
        # 0の近似値(値がZERO以下なら，
        # 対数はLZEROに置き換える)
        self.ZERO = 1E-100
        # 分散値のフロアリング値
        self.MINVAR = 1E-4

        #
        # 学習時/認識時に使うパラメータ
        #
        # 正規分布毎に計算される対数確率
        self.elem_prob = None
        # 状態毎に計算される対数確率
        self.state_prob = None
        # 前向き確率
        self.alpha = None
        # 後ろ向き確率
        self.beta = None
        # HMM尤度
        self.loglikelihood = 0
        # パラメータ更新のための変数
        self.pdf_accumulators = None
        self.trans_accumulators = None
        # ビタビアルゴリズム時に用いる累積確率
        self.score = None
        # ビタビパスを記憶する行列
        self.track = None
        # ビタビアルゴリズムによるスコア
        self.viterbi_score = 0

    def make_proto(self,
                   phone_list,
                   num_states,
                   prob_loop,
                   num_dims):
        ''' HMMのプロトタイプを作成する
        phone_list: 音素のリスト
        num_states: 各音素HMMの状態数
        prob_loop:  自己ループ確率
        num_dims:   特徴量の次元数
        '''
        # 音素リストを得る
        self.phones = phone_list
        # 音素数を得る
        self.num_phones = len(self.phones)
        # 各音素HMMの状態数を得る
        self.num_states = num_states
        # 特徴量ベクトルの次元数を得る
        self.num_dims = num_dims
        # GMMの混合数は1とする
        self.num_mixture = 1
              
        # 正規分布を作成
        # 音素番号p, 状態番号s, 混合要素番号m
        # の正規分布はpdf[p][s][m]でアクセスする
        # pdf[p][s][m] = gaussian
        self.pdf = []
        for p in range(self.num_phones):
            tmp_p = []
            for s in range(self.num_states):
                tmp_s = []
                for m in range(self.num_mixture):
                    # 平均値ベクトルを定義
                    mu = np.zeros(self.num_dims)
                    # 対角共分散行列の対角成分を定義
                    var = np.ones(self.num_dims)
                    # 混合数は1なので，混合重みは1.0
                    weight = 1.0
                    # gConst項を計算
                    gconst = self.calc_gconst(var)
                    # 正規分布を辞書型で定義
                    gaussian = {'weight': weight, 
                                'mu': mu, 
                                'var': var,
                                'gConst': gconst}
                    # 正規分布を加える          
                    tmp_s.append(gaussian)
                tmp_p.append(tmp_s)
            self.pdf.append(tmp_p)

        # 状態遷移確率(の対数値)を作成
        # 音素番号p, 状態番号s の遷移確率は
        # trans[p][s] = [loop, next]
        # loop: 自己ループ確率
        # next: 次の状態への遷移確率

        # 次の状態へ遷移する確率
        prob_next = 1.0 - prob_loop
        # 対数を取る
        log_prob_loop = np.log(prob_loop) \
            if prob_loop > self.ZERO else self.LZERO
        log_prob_next = np.log(prob_next) \
            if prob_next > self.ZERO else self.LZERO

        # self.transに格納していく
        self.trans = []
        for p in range(self.num_phones):
            tmp_p = []
            for s in range(self.num_states):
                tmp_trans = np.array([log_prob_loop, 
                                      log_prob_next])
                tmp_p.append(tmp_trans)
            self.trans.append(tmp_p)


    def calc_gconst(self, variance):
        ''' gConst項(正規分布の定数項
            の対数値)を計算する
        variance: 対角共分散行列の対角成分
        '''       
        gconst = self.num_dims * np.log(2.0 * np.pi) \
               + np.sum(np.log(variance))
        return gconst


    def calc_pdf(self, pdf, obs):
        ''' 指定した正規分布での対数尤度を計算
        pdf:     正規分布
        obs:     入力特徴量
                 1フレーム分のベクトルでも
                 フレームx次元の配列でも入力可能
        logprob: 対数尤度
                 1フレーム分を与えた場合はスカラ値
                 複数フレーム分与えた場合は
                 フレーム数分のサイズを持つベクトル
        '''
        # 定数項を除く部分の計算(exp(*)の部分)
        tmp = (obs - pdf['mu'])**2 / pdf['var']
        if np.ndim(tmp) == 2:
            # obsが[フレームx次元]の配列で入力された場合
            tmp = np.sum(tmp, 1)
        elif np.ndim(tmp) == 1:
            # obsが1フレーム分のベクトルで入力された場合
            tmp = np.sum(tmp)
        # 定数項をつけて-0.5をかける
        logprob = -0.5 * (tmp + pdf['gConst'])
        return logprob


    def logadd(self, x, y):
        ''' x=log(a)とy=log(b)に対して
            log(a+b)を計算する
        x: log(a)
        y: log(b)
        z: log(a+b)
        '''
        if x > y:
            z = x + np.log(1.0 + np.exp(y - x))
        else:
            z = y + np.log(1.0 + np.exp(x - y))
        return z


    def flat_init(self, mean, var):
        ''' フラットスタートによる初期化
            学習データ全体の平均分散を
            HMMの全正規分布のパラメータにする
        mean: 学習データ全体の平均ベクトル
        var:  学習データ全体の対角共分散
        '''
        # 次元数が一致しない場合はエラー
        if self.num_dims != len(mean) or \
           self.num_dims != len(var):
            sys.stderr.write('flat_init: invalid mean or var\n')
            return 1
        for p in range(self.num_phones):
            for s in range(self.num_states):
                for m in range(self.num_mixture):
                    pdf = self.pdf[p][s][m]
                    pdf['mu'] = mean
                    pdf['var'] = var
                    pdf['gConst'] = self.calc_gconst(var)


    def calc_out_prob(self, feat, label):
        ''' 出力確率の計算
        feat: 1発話分の特徴量[フレーム数x次元数]
        label 1発話分のラベル
        '''
        # 特徴量のフレーム数を得る
        feat_len = np.shape(feat)[0]
        # ラベルの長さを得る
        label_len = len(label)

        # 正規分布毎に計算される対数確率
        self.elem_prob = np.zeros((label_len, 
                                   self.num_states, 
                                   self.num_mixture, 
                                   feat_len))

        # 各状態(q,s)における時刻tの出力確率
        # (state_prob = sum(weight*elem_prob))
        self.state_prob = np.zeros((label_len,
                                    self.num_states,
                                    feat_len))

        # elem_prob, state_prob を計算していく
        # l: ラベル上の何番目の音素か
        # p: lが音素リスト上のどの音素か
        # s: 状態
        # t: フレーム
        # m: 混合要素
        for l, p in enumerate(label):
            for s in range(self.num_states):
                # state_probをlog(0)で初期化
                self.state_prob[l][s][:] = \
                    self.LZERO * np.ones(feat_len)
                for m in range(self.num_mixture):
                    # 正規分布を取り出す
                    pdf = self.pdf[p][s][m]
                    # 確率の計算
                    self.elem_prob[l][s][m][:] = \
                        self.calc_pdf(pdf, feat)
                    # GMMの重みを加える
                    tmp_prob = np.log(pdf['weight']) \
                        + self.elem_prob[l][s][m][:]
                    # 確率を足す
                    for t in range(feat_len):
                        self.state_prob[l][s][t] = \
                            self.logadd(self.state_prob[l][s][t],
                                        tmp_prob[t])


    def calc_alpha(self, label):
        ''' 前向き確率alphaを求める
            left-to-right型HMMを前提とした実装に
            なっている
        label: ラベル
        '''
        # ラベル長とフレーム数を得る
        (label_len, _, feat_len) = np.shape(self.state_prob)
        # alphaをlog(0)で初期化
        self.alpha = self.LZERO * np.ones((label_len,
                                           self.num_states,
                                           feat_len))
        
        # t=0のとき，
        # 必ず最初の音素の最初の状態にいる
        self.alpha[0][0][0] = self.state_prob[0][0][0]

        # t: フレーム       
        # l: ラベル上の何番目の音素か
        # p: lが音素リスト上のどの音素か
        # s: 状態
        for t in range(1, feat_len):
            for l in range(0, label_len):
                p = label[l]
                for s in range(0, self.num_states):
                    # 自己ループの考慮
                    self.alpha[l][s][t] = \
                        self.alpha[l][s][t-1] \
                        + self.trans[p][s][0]
                    if s > 0:
                        # 先頭の状態でなければ，
                        # 一つ前の状態からの遷移を考慮
                        tmp = self.alpha[l][s-1][t-1] \
                            + self.trans[p][s-1][1]
                        # 自己ループとの和を計算
                        if tmp > self.LSMALL:
                            self.alpha[l][s][t] = \
                                self.logadd(self.alpha[l][s][t], 
                                            tmp)
                    elif l > 0:
                        # 先頭の音素ではなく，
                        # かつ先頭の状態の場合
                        # 一つ前の音素の終端状態から遷移
                        prev_p = label[l-1]
                        tmp = self.alpha[l-1][-1][t-1] \
                            + self.trans[prev_p][-1][1]
                        # 自己ループとの和を計算
                        if tmp > self.LSMALL:
                            self.alpha[l][s][t] = \
                                self.logadd(self.alpha[l][s][t], 
                                            tmp)
                    # else:
                    #   # 先頭の音素かつ先頭の状態の場合
                    #   # 自己ループ以外の遷移は無い

                    # state_probを加える
                    self.alpha[l][s][t] += \
                        self.state_prob[l][s][t]

        # HMMの対数尤度は終端のalpha
        self.loglikelihood = self.alpha[-1][-1][-1]

    def calc_beta(self, label):
        ''' 後ろ向き確率betaを求める
            left-to-right型HMMを前提とした実装に
            なっている
        label: ラベル
        '''
        # ラベル長とフレーム数を得る
        (label_len, _, feat_len) = np.shape(self.state_prob)
        # alphaをlog(0)で初期化
        self.beta = self.LZERO * np.ones((label_len,
                                          self.num_states,
                                          feat_len))
        
        # t=-1 (最終フレーム)のとき，
        # 必ず最後の音素の最後の状態にいる
        # (確率はlog(1) = 0)
        self.beta[-1][-1][-1] = 0.0

        # t: フレーム       
        # l: ラベル上の何番目の音素か
        # p: lが音素リスト上のどの音素か
        # s: 状態
        # calc_alphaと違い，tはfeat_len-2から0へ
        # 進む点に注意
        for t in range(0, feat_len-1)[::-1]:
            for l in range(0, label_len):
                p = label[l]
                for s in range(0, self.num_states):
                    # 自己ループの考慮
                    self.beta[l][s][t] = \
                        self.beta[l][s][t+1] \
                        + self.trans[p][s][0] \
                        + self.state_prob[l][s][t+1]
                    if s < self.num_states - 1:
                        # 終端の状態でなければ，
                        # 一つ後の状態への遷移を考慮
                        tmp = self.beta[l][s+1][t+1] \
                            + self.trans[p][s][1] \
                            + self.state_prob[l][s+1][t+1]
                        # 自己ループとの和を計算
                        if tmp > self.LSMALL:
                            self.beta[l][s][t] = \
                                self.logadd(self.beta[l][s][t], 
                                            tmp)
                    elif l < label_len - 1:
                        # 終端の音素ではなく，
                        # かつ終端の状態の場合
                        # 一つ後の音素の先頭の状態への遷移
                        tmp = self.beta[l+1][0][t+1] \
                            + self.trans[p][s][1] \
                            + self.state_prob[l+1][0][t+1]
                        # 自己ループとの和を計算
                        if tmp > self.LSMALL:
                            self.beta[l][s][t] = \
                                self.logadd(self.beta[l][s][t], 
                                            tmp)
                    # else:
                    #   # 終端の音素かつ終端の状態の場合
                    #   # 自己ループ以外の遷移は無い


    def reset_accumulators(self):
        ''' accumulators (パラメータ更新に必要な変数)
            を初期化する
        '''
        # GMMを更新するためのaccumulators
        self.pdf_accumulators = []
        for p in range(self.num_phones):
            tmp_p = []
            for s in range(self.num_states):
                tmp_s = []
                for m in range(self.num_mixture):
                    pdf_stats = {}
                    pdf_stats['weight'] = \
                        {'num': self.LZERO, 
                         'den': self.LZERO}
                    pdf_stats['mu'] = \
                        {'num': np.zeros(self.num_dims),
                         'den': self.LZERO}
                    pdf_stats['var'] = \
                        {'num': np.zeros(self.num_dims),
                         'den': self.LZERO}
                    tmp_s.append(pdf_stats)
                tmp_p.append(tmp_s)
            self.pdf_accumulators.append(tmp_p)
        
        # 遷移確率を更新するためのaccumulators
        self.trans_accumulators = []
        for p in range(self.num_phones):
            tmp_p = []
            for s in range(self.num_states):
                trans_stats = \
                    {'num': np.ones(2) * self.LZERO, 
                     'den': self.LZERO}
                tmp_p.append(trans_stats)
            self.trans_accumulators.append(tmp_p)


    def update_accumulators(self, feat, label):
        ''' accumulatorsの更新
            left-to-rightを前提とした実装になっている
        feat: 特徴量
        label: ラベル
        '''
        # ラベルの長さを取得
        label_len = len(label)
        # フレーム数を取得
        feat_len = np.shape(feat)[0]

        # t: フレーム       
        # l: ラベル上の何番目の音素か
        # p: lが音素リスト上のどの音素か
        # s: 状態
        for t in range(feat_len):
            for l in range(label_len):
                p = label[l]
                for s in range(self.num_states):
                    if t == 0 and l == 0 and s == 0:
                        # t=0の時は必ず先頭の状態
                        # (対数確率なのでlog(1)=0)
                        lconst = 0
                    elif t == 0:
                        # t=0で先頭の状態でない場合は
                        # 確率ゼロなのでスキップ
                        continue   
                    elif s > 0:
                        # t>0 で先頭の状態でない場合
                        # 自己ループ
                        lconst = self.alpha[l][s][t-1] \
                               + self.trans[p][s][0]
                        # 一つ前の状態からの遷移を考慮
                        tmp = self.alpha[l][s-1][t-1] \
                            + self.trans[p][s-1][1]
                        # 自己ループとの和を計算
                        if tmp > self.LSMALL:
                            lconst = self.logadd(lconst, 
                                                 tmp)
                    elif l > 0:
                        # t>0 先頭の音素ではなく，
                        # かつ先頭の状態の場合
                        # 自己ループ
                        lconst = self.alpha[l][s][t-1] \
                               + self.trans[p][s][0]
                        # 一つ前の音素の終端状態から遷移
                        prev_p = label[l-1]
                        tmp = self.alpha[l-1][-1][t-1] \
                            + self.trans[prev_p][-1][1]
                        # 自己ループとの和を計算
                        if tmp > self.LSMALL:
                            lconst = self.logadd(lconst, 
                                                 tmp)
                    else:
                        # 先頭の音素かつ先頭の状態の場合
                        # 自己ループのみ
                        lconst = self.alpha[l][s][t-1] \
                               + self.trans[p][s][0]
                    
                    # 後ろ向き確率と1/Pを加える
                    lconst += self.beta[l][s][t] \
                           - self.loglikelihood
                    # accumulatorsの更新
                    for m in range(self.num_mixture):
                        pdf = self.pdf[p][s][m]
                        L = lconst \
                          + np.log(pdf['weight']) \
                          + self.elem_prob[l][s][m][t]
                        
                        pdf_accum = self.pdf_accumulators[p][s][m]
                        # 平均値ベクトル更新式の分子は
                        # 対数を取らない
                        pdf_accum['mu']['num'] += \
                            np.exp(L) * feat[t]
                        # 分母は対数上で更新
                        if L > self.LSMALL:
                            pdf_accum['mu']['den'] = \
                                self.logadd(pdf_accum['mu']['den'], 
                                            L)
                        # 対角共分散更新式の分子は
                        # 対数を取らない
                        dev = feat[t] - pdf['mu']
                        pdf_accum['var']['num'] += \
                            np.exp(L) * (dev**2)
                        # 分母は平均値のものと同じ値
                        pdf_accum['var']['den'] = \
                            pdf_accum['mu']['den']

                        # GMM重み更新式の分子は
                        # 平均・分散の分母と同じ値
                        pdf_accum['weight']['num'] = \
                            pdf_accum['mu']['den']

        # 遷移確率のaccumulatorsと
        # GMM重みのaccumulatorsの分母を更新
        for t in range(feat_len):
            for l in range(label_len):
                p = label[l]
                for s in range(self.num_states):
                    # GMM重みaccumulatorの分母と
                    # 遷移確率accumulatorの分母の更新に用いる
                    alphabeta = self.alpha[l][s][t] \
                              + self.beta[l][s][t] \
                              - self.loglikelihood

                    # GMM重みaccumulatorの分母を更新
                    for m in range(self.num_mixture):
                        pdf_accum = \
                            self.pdf_accumulators[p][s][m]
                        # 分母は全てのmで同じ値なので、
                        # m==0のときのみ計算
                        if m == 0:
                            if alphabeta > self.LSMALL:
                                pdf_accum['weight']['den'] = \
                                    self.logadd(\
                                        pdf_accum['weight']['den'],
                                        alphabeta)
                        else:
                            tmp = self.pdf_accumulators[p][s][0]
                            pdf_accum['weight']['den'] = \
                                tmp['weight']['den']

                    # 遷移確率accumulatorの分母を更新
                    trans_accum = self.trans_accumulators[p][s]
                    if t < feat_len - 1 \
                            and alphabeta > self.LSMALL:
                        trans_accum['den'] = \
                            self.logadd(trans_accum['den'],
                                        alphabeta)

                    #
                    # 以下は遷移確率accumulatorの分子の更新
                    #
                    if t == feat_len - 1:
                        # 最終フレームはスキップ
                        continue
                    elif s < self.num_states - 1:
                        # 各音素の非終端状態の場合
                        # 自己ループ
                        tmp = self.alpha[l][s][t] \
                            + self.trans[p][s][0] \
                            + self.state_prob[l][s][t+1] \
                            + self.beta[l][s][t+1] \
                            - self.loglikelihood
                        if tmp > self.LSMALL:
                            trans_accum['num'][0] = \
                                self.logadd(trans_accum['num'][0],
                                            tmp)

                        # 遷移
                        tmp = self.alpha[l][s][t] \
                            + self.trans[p][s][1] \
                            + self.state_prob[l][s+1][t+1] \
                            + self.beta[l][s+1][t+1] \
                            - self.loglikelihood
                        if tmp > self.LSMALL:
                            trans_accum['num'][1] = \
                                self.logadd(trans_accum['num'][1],
                                            tmp)
                    elif l < label_len - 1:
                        # 終端状態かつ非終端音素
                        # 自己ループ
                        tmp = self.alpha[l][s][t] \
                            + self.trans[p][s][0] \
                            + self.state_prob[l][s][t+1] \
                            + self.beta[l][s][t+1] \
                            - self.loglikelihood
                        if tmp > self.LSMALL:
                            trans_accum['num'][0] = \
                                self.logadd(trans_accum['num'][0],
                                            tmp)
                        # 次の音素の始端状態への遷移
                        tmp = self.alpha[l][s][t] \
                            + self.trans[p][s][1] \
                            + self.state_prob[l+1][0][t+1] \
                            + self.beta[l+1][0][t+1] \
                            - self.loglikelihood
                        if tmp > self.LSMALL:
                            trans_accum['num'][1] = \
                                self.logadd(trans_accum['num'][1],
                                            tmp)
                    else:
                        # 最終状態
                        # 自己ループ
                        tmp = self.alpha[l][s][t] \
                            + self.trans[p][s][0] \
                            + self.state_prob[l][s][t+1] \
                            + self.beta[l][s][t+1] \
                            - self.loglikelihood
                        if tmp > self.LSMALL:
                            trans_accum['num'][0] = \
                                self.logadd(trans_accum['num'][0],
                                            tmp)


    def update_parameters(self):
        ''' パラメータの更新
        '''
        for p in range(self.num_phones):
            for s in range(self.num_states):
                # 遷移確率の更新
                trans_accum = self.trans_accumulators[p][s]
                self.trans[p][s] = \
                    trans_accum['num'] - trans_accum['den']
                # 確率総和が1になるよう正規化
                tmp = self.logadd(self.trans[p][s][0], 
                                  self.trans[p][s][1])
                self.trans[p][s] -= tmp
                for m in range(self.num_mixture):
                    pdf = self.pdf[p][s][m]
                    pdf_accum = self.pdf_accumulators[p][s][m]
                    # 平均値ベクトルの更新
                    den = np.exp(pdf_accum['mu']['den'])
                    if den > 0:
                        pdf['mu'] = pdf_accum['mu']['num'] / den
                    # 対角共分散の更新
                    den = np.exp(pdf_accum['var']['den'])
                    if den > 0:
                        pdf['var'] = pdf_accum['var']['num'] / den
                    # 分散のフロアリング
                    pdf['var'][pdf['var'] < self.MINVAR] = \
                        self.MINVAR
                    # gConst項の更新
                    gconst = self.calc_gconst(pdf['var'])
                    pdf['gConst'] = gconst

                    # GMM重みの更新
                    tmp = pdf_accum['weight']['num'] - \
                        pdf_accum['weight']['den']
                    pdf['weight'] = np.exp(tmp)
                # GMM重みの総和が1になるよう正規化    
                wsum = 0.0
                for m in range(self.num_mixture):
                    wsum += self.pdf[p][s][m]['weight']
                for m in range(self.num_mixture):
                    self.pdf[p][s][m]['weight'] /= wsum


    def viterbi_decoding(self, label):
        ''' ビタビアルゴリズムによるデコーディング
            left-to-right型HMMを前提とした実装に
            なっている
        lable: ラベル
        '''
        # ラベル長とフレーム数を得る
        (label_len, _, feat_len) = np.shape(self.state_prob)
        # scoreをlog(0)で初期化
        self.score = self.LZERO * np.ones((label_len,
                                          self.num_states,
                                          feat_len))
        # バックトラック用の遷移記録領域
        # 0:自己ループ 1:次の状態に遷移
        self.track = np.zeros((label_len,
                               self.num_states,
                               feat_len), np.int16)
        # t=0のとき，
        # 必ず最初の音素の最初の状態にいる
        self.score[0][0][0] = self.state_prob[0][0][0]

        # t: フレーム       
        # l: ラベル上の何番目の音素か
        # p: lが音素リスト上のどの音素か
        # s: 状態
        for t in range(1, feat_len):
            for l in range(0, label_len):
                p = label[l]
                for s in range(0, self.num_states):
                    if s > 0:
                        # 先頭の状態でなければ，
                        # 一つ前の状態からの遷移か
                        # 自己ループのいずれか
                        p_next = self.score[l][s-1][t-1] \
                               + self.trans[p][s-1][1]
                        p_loop = self.score[l][s][t-1] \
                               + self.trans[p][s][0]
                        # 大きい方を採用
                        cand = [p_loop, p_next]
                        tran = np.argmax(cand)
                        self.score[l][s][t] = cand[tran]
                        self.track[l][s][t] = tran
                    elif l > 0:
                        # 先頭の音素ではなく，
                        # かつ先頭の状態の場合
                        # 一つ前の音素の終端状態から遷移か
                        # 自己ループのいずれか
                        prev_p = label[l-1]
                        p_next = self.score[l-1][-1][t-1] \
                               + self.trans[prev_p][-1][1]
                        p_loop = self.score[l][s][t-1] \
                               + self.trans[p][s][0]
                        # 大きい方を採用
                        cand = [p_loop, p_next]
                        tran = np.argmax(cand)
                        self.score[l][s][t] = cand[tran]
                        self.track[l][s][t] = tran
                    else:
                        # 先頭の音素かつ先頭の状態の場合
                        # 自己ループのみ
                        p_loop = self.score[l][s][t-1] \
                               + self.trans[p][s][0]
                        self.score[l][s][t] = p_loop
                        self.track[l][s][t] = 0

                    # state_probを加える
                    self.score[l][s][t] += \
                        self.state_prob[l][s][t]

        # ビタビスコア終端のscore
        self.viterbi_score = self.score[-1][-1][-1]


    def back_track(self):
        ''' ビタビパスのバックトラック
        viterbi_path: バックトラックの結果
        '''
        # ラベル長とフレーム数を得る
        (label_len, _, feat_len) = np.shape(self.track)
 
        viterbi_path = []
        # 終端からスタート
        l = label_len - 1       # 音素
        s = self.num_states - 1 # 状態
        t = feat_len - 1        # フレーム
        while True:
            viterbi_path.append([l, s, t])
            # スタート地点に到達したら終了
            if l == 0 and s == 0 and t == 0:
                break
            # trackの値を見る
            # 0なら自己ループ，1なら遷移
            tran = self.track[l][s][t]
            
            if tran == 1:
                # 遷移
                if s == 0:
                    # 前の音素からの遷移
                    # lを減らしてsを終端にする
                    l = l - 1
                    s = self.num_states - 1
                else:
                    # 同じ音素の前の状態からの遷移
                    # sを減らす
                    s = s - 1
            # tを減らす
            t = t - 1

        # viterbi_path を逆順に並び替える
        viterbi_path = viterbi_path[::-1]
        return viterbi_path


    def mixup(self):
        ''' HMMの混合数を2倍に増やす
        '''
              
        for p in range(self.num_phones):
            for s in range(self.num_states):
                pdf = self.pdf[p][s]
                for m in range(self.num_mixture):
                    # 混合重みを得る
                    weight = pdf[m]['weight']
                    # 混合数を2倍に増やした分重みを0.5倍する
                    weight *= 0.5
                    # コピー元の混合重みも0.5倍する
                    pdf[m]['weight'] *= 0.5
                    # gConst項を得る
                    gconst = pdf[m]['gConst']

                    # 平均値ベクトルを得る
                    mu = pdf[m]['mu'].copy()
                    # 対角共分散を得る
                    var = pdf[m]['var'].copy()

                    # 標準偏差を得る
                    std = np.sqrt(var)                  
                    # 標準偏差の0.2倍を平均値ベクトルに足す
                    mu = mu + 0.2 * std
                    # コピー元の平均値ベクトルは0.2*stdで引く
                    pdf[m]['mu'] = pdf[m]['mu'] - 0.2*std

                    # 正規分布を辞書型で定義
                    gaussian = {'weight': weight, 
                                'mu': mu, 
                                'var': var,
                                'gConst': gconst}
                    # 正規分布を加える          
                    pdf.append(gaussian)

        # GMMの混合数を2倍にする
        self.num_mixture *= 2


    def train(self, feat_list, label_list, report_interval=10):
        ''' HMMを 1 iteration分 更新する
        feat_list:  特徴量ファイルのリスト
                    発話IDをkey，特徴量ファイルパスを
                    valueとする辞書
        label_list: ラベルのリスト
                    発話IDをkey，ラベルをvalueとする
                    辞書
        report_interval: 処理途中結果を表示する間隔(発話数)
        '''
        # accumulators(パラメータ更新に
        # 用いる変数)をリセット
        self.reset_accumulators()

        # 特徴量ファイルを一つずつ開いて処理
        count = 0
        ll_per_utt = 0.0
        partial_ll = 0.0
        for utt, ff in feat_list.items():
            # 処理した発話数を1増やす
            count += 1
            # 特徴量ファイルを開く
            feat = np.fromfile(ff, dtype=np.float32)
            # フレーム数 x 次元数の配列に変形
            feat = feat.reshape(-1, self.num_dims)
            # ラベルを取得
            label = label_list[utt]

            # 各分布の出力確率を求める
            self.calc_out_prob(feat, label)
            # 前向き確率を求める
            self.calc_alpha(label)
            # 後ろ向き確率を求める
            self.calc_beta(label)
            # accumulatorsを更新する
            self.update_accumulators(feat, label)
            # 対数尤度を加算
            ll_per_utt += self.loglikelihood

            # 途中結果を表示する
            partial_ll += self.loglikelihood
            if count % report_interval == 0:
                partial_ll /= report_interval
                print('  %d / %d utterances processed' \
                      % (count, len(feat_list)))
                print('  log likelihood averaged'\
                      ' over %d utterances: %f' \
                      % (report_interval, partial_ll))

        # モデルパラメータを更新
        self.update_parameters()
        #対数尤度の発話平均を求める
        ll_per_utt /= count
        print('average log likelihood: %f' % (ll_per_utt))


    def recognize(self, feat, lexicon):
        ''' 孤立単語認識を行う
        feat:    特徴量
        lexicon: 認識単語リスト．
                 以下の辞書型がリストになっている．
                 {'word':単語, 
                  'pron':音素列,
                  'int':音素列の数値表記}
        '''
        # 単語リスト内の単語毎に尤度を計算する
        # 結果リスト
        result = []
        for lex in lexicon:
            # 音素列の数値表記を得る
            label = lex['int']
            # 各分布の出力確率を求める
            self.calc_out_prob(feat, label)
            # ビタビアルゴリズムを実行
            self.viterbi_decoding(label)
            result.append({'word': lex['word'],
                           'score': self.viterbi_score})

        # スコアの昇順に並び替える
        result = sorted(result, 
                        key=lambda x:x['score'], 
                        reverse=True)
        # 認識結果とスコア情報を返す
        return (result[0]['word'], result)


    def set_out_prob(self, prob, label):
        ''' 出力確率をセットする
        prob DNNが出力する確率を想定
             [フレーム数 x (音素数*状態数)]
             の2次元配列になっている
        label 1発話分のラベル
        '''
        # フレーム数を得る
        feat_len = np.shape(prob)[0]
        # ラベルの長さを得る
        label_len = len(label)

        # 各状態(q,s)における時刻tの出力確率
        # (state_prob = sum(weight*elem_prob))
        self.state_prob = np.zeros((label_len,
                                    self.num_states,
                                    feat_len))

        # state_prob を計算していく
        # l: ラベル上の何番目の音素か
        # p: lが音素リスト上のどの音素か
        # s: 状態
        # t: フレーム
        for l, p in enumerate(label):
            for s in range(self.num_states):
                # 音素pの状態sの値はDNN出力上で
                # p*num_states+s に格納されている
                state = p * self.num_states + s
                for t in range(feat_len):
                    self.state_prob[l][s][t] = \
                        prob[t][state]


    def recognize_with_dnn(self, prob, lexicon):
        ''' DNNの出力した確率値を用いて
            孤立単語認識を行う
        prob:    DNNの出力確率
                 (ただし各状態の事前確率で割って
                 尤度に変換しておくこと)
        lexicon: 認識単語リスト．
                 以下の辞書型がリストになっている．
                 {'word':単語, 
                  'pron':音素列,
                  'int':音素列の数値表記}
        '''
        # 単語リスト内の単語毎に尤度を計算する
        # 結果リスト
        result = []
        for lex in lexicon:
            # 音素列の数値表記を得る
            label = lex['int']
            # 各分布の出力確率をセットする
            self.set_out_prob(prob, label)
            # ビタビアルゴリズムを実行
            self.viterbi_decoding(label)
            result.append({'word': lex['word'],
                           'score': self.viterbi_score})

        # スコアの昇順に並び替える
        result = sorted(result, 
                        key=lambda x:x['score'], 
                        reverse=True)
        # 認識結果とスコア情報を返す
        return (result[0]['word'], result)


    def phone_alignment(self, feat, label):
        ''' 音素アライメントを行う
        feat: 特徴量
        label: ラベル
        '''
        # 各分布の出力確率を求める
        self.calc_out_prob(feat, label)
        # ビタビアルゴリズムを実行
        self.viterbi_decoding(label)
        # バックトラックを実行
        viterbi_path = self.back_track()
        # ビタビパスからフレーム毎の音素列に変換
        phone_alignment = []
        for vp in viterbi_path:
            # ラベル上の音素インデクスを取得
            l = vp[0]
            # 音素番号を音素リスト上の番号に変換
            p = label[l]
            # 番号から音素記号に変換
            ph = self.phones[p]
            # phone_alignmentの末尾に追加
            phone_alignment.append(ph)

        return phone_alignment


    def state_alignment(self, feat, label):
        ''' HMM状態でのアライメントを行う
        feat: 特徴量
        label: ラベル
        state_alignment: フレーム毎の状態番号
            ただしここでの状態番号は
            (音素番号)*(状態数)+(音素内の状態番号)
            とする．
        '''
        # 各分布の出力確率を求める
        self.calc_out_prob(feat, label)
        # ビタビアルゴリズムを実行
        self.viterbi_decoding(label)
        # バックトラックを実行
        viterbi_path = self.back_track()
        # ビタビパスからフレーム毎の状態番号列に変換
        state_alignment = []
        for vp in viterbi_path:
            # ラベル上の音素インデクスを取得
            l = vp[0]
            # 音素番号を音素リスト上の番号に変換
            p = label[l]
            # 音素内の状態番号を取得
            s = vp[1]
            # 出力時の状態番号は
            # p * num_states + s とする
            state = p * self.num_states + s
            # phone_alignmentの末尾に追加
            state_alignment.append(state)

        return state_alignment


    def save_hmm(self, filename):
        ''' HMMパラメータをjson形式で保存
        filename: 保存ファイル名
        '''
        # json形式で保存するため，
        # HMMの情報を辞書形式に変換する
        hmmjson = {}
        # 基本情報を入力
        hmmjson['num_phones'] = self.num_phones
        hmmjson['num_states'] = self.num_states
        hmmjson['num_mixture'] = self.num_mixture
        hmmjson['num_dims'] = self.num_dims
        # 音素モデルリスト
        hmmjson['hmms'] = []
        for p, phone in enumerate(self.phones):
            model_p = {}
            # 音素名
            model_p['phone'] = phone
            # HMMリスト
            model_p['hmm'] = []
            for s in range(self.num_states):
                model_s = {}
                # 状態番号
                model_s['state'] = s
                # 遷移確率(対数値から戻す)
                model_s['trans'] = \
                    list(np.exp(self.trans[p][s]))
                # GMMリスト
                model_s['gmm'] = []
                for m in range(self.num_mixture):
                    model_m = {}
                    # 混合要素番号
                    model_m['mixture'] = m
                    # 混合重み
                    model_m['weight'] = \
                        self.pdf[p][s][m]['weight']
                    # 平均値ベクトル
                    # jsonはndarrayを扱えないので
                    # list型に変換しておく
                    model_m['mean'] = \
                        list(self.pdf[p][s][m]['mu'])
                    # 対角共分散
                    model_m['variance'] = \
                        list(self.pdf[p][s][m]['var'])
                    # gConst
                    model_m['gConst'] = \
                        self.pdf[p][s][m]['gConst']
                    # gmmリストに加える
                    model_s['gmm'].append(model_m)
                # hmmリストに加える
                model_p['hmm'].append(model_s)
            # 音素モデルリストに加える
            hmmjson['hmms'].append(model_p)

        # JSON形式で保存する
        with open(filename, mode='w') as f:
            json.dump(hmmjson, f, indent=4)


    def load_hmm(self, filename):
        ''' json形式のHMMファイルを読み込む
        filename: 読み込みファイル名
        '''
        # JSON形式のHMMファイルを読み込む
        with open(filename, mode='r') as f:
            hmmjson = json.load(f)

        # 辞書の値を読み込んでいく
        self.num_phones = hmmjson['num_phones']
        self.num_states = hmmjson['num_states']
        self.num_mixture = hmmjson['num_mixture']
        self.num_dims = hmmjson['num_dims']

        # 音素情報の読み込み
        self.phones = []
        for p in range(self.num_phones):
            hmms = hmmjson['hmms'][p]
            self.phones.append(hmms['phone'])

        #遷移確率の読み込み
        # 音素番号p, 状態番号s の遷移確率は
        # trans[p][s] = [loop, next]
        self.trans = []
        for p in range(self.num_phones):
            tmp_p = []
            hmms = hmmjson['hmms'][p]
            for s in range(self.num_states):
                hmm = hmms['hmm'][s]
                # 遷移確率の読み込み
                tmp_trans = np.array(hmm['trans'])
                # 総和が1になるよう正規化
                tmp_trans /= np.sum(tmp_trans)
                # 対数に変換
                for i in [0, 1]:
                    tmp_trans[i] = np.log(tmp_trans[i]) \
                        if tmp_trans[i] > self.ZERO \
                        else self.LZERO
                tmp_p.append(tmp_trans)
            # self.transに追加
            self.trans.append(tmp_p)

        # 正規分布パラメータの読み込み
        # 音素番号p, 状態番号s, 混合要素番号m
        # の正規分布はpdf[p][s][m]でアクセスする
        # pdf[p][s][m] = gaussian
        self.pdf = []
        for p in range(self.num_phones):
            tmp_p = []
            hmms = hmmjson['hmms'][p]
            for s in range(self.num_states):
                tmp_s = []
                hmm = hmms['hmm'][s]
                for m in range(self.num_mixture):
                    gmm = hmm['gmm'][m]
                    # 重み，平均，分散，gConstを取得
                    weight = gmm['weight']
                    mu = np.array(gmm['mean'])
                    var = np.array(gmm['variance'])
                    gconst = gmm['gConst']
                    # 正規分布を作成
                    gaussian = {'weight': weight, 
                                'mu': mu, 
                                'var': var,
                                'gConst': gconst}
                    tmp_s.append(gaussian)
                tmp_p.append(tmp_s)
            # self.pdfに追加
            self.pdf.append(tmp_p)

