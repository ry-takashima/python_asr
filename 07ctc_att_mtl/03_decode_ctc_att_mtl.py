# -*- coding: utf-8 -*-

#
# RNN Attention Encoder-Decoderによるデコーディングを行います
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
from torch.utils.data import DataLoader

# 作成したDatasetクラスをインポート
from my_dataset import SequenceDataset

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# モデルの定義をインポート
from my_model import MyMTLModel

# json形式の入出力を行うモジュールをインポート
import json

# os, sysモジュールをインポート
import os
import sys


#
# メイン関数
#
if __name__ == "__main__":
    
    #
    # 設定ここから
    #

    # トークンの単位
    # phone:音素  kana:かな  char:キャラクター
    unit = 'phone'

    # Multi Task TrainingのCTC損失関数重み
    mtl_weight = 0.5

    # 実験ディレクトリ
    exp_dir = './exp_train_small'

    # 評価データの特徴量(feats.scp)が存在するディレクトリ
    feat_dir_test = '../01compute_features/fbank/test'

    # 評価データの特徴量リストファイル
    feat_scp_test = os.path.join(feat_dir_test, 'feats.scp')

    # 評価データのラベルファイル
    label_test = os.path.join(exp_dir, 'data', unit, 'label_test')

    # トークンリスト
    token_list_path = os.path.join(exp_dir, 'data', unit,
                                   'token_list')

    # 学習済みモデルが格納されているディレクトリ
    model_dir = os.path.join(exp_dir,
                             unit+'_model_attention_mtl'\
                             +str(mtl_weight))

    # 訓練データから計算された特徴量の平均/標準偏差ファイル
    mean_std_file = os.path.join(model_dir, 'mean_std.txt')

    # 学習済みのモデルファイル
    model_file = os.path.join(model_dir, 'best_model.pt')

    # デコード結果を出力するディレクトリ
    output_dir = os.path.join(model_dir, 'decode_test')

    # デコード結果および正解文の出力ファイル
    hypothesis_file = os.path.join(output_dir, 'hypothesis.txt')
    reference_file = os.path.join(output_dir, 'reference.txt')

    # 学習時に出力した設定ファイル
    config_file = os.path.join(model_dir, 'config.json')

    # ミニバッチに含める発話数
    batch_size = 10
    
    #
    # 設定ここまで
    #

    # 設定ファイルを読み込む
    with open(config_file, mode='r') as f:
        config = json.load(f)

    # 読み込んだ設定を反映する
    # Encoderの設定
    # 中間層のレイヤー数
    enc_num_layers = config['enc_num_layers']
    # 層ごとのsub sampling設定
    enc_sub_sample = config['enc_sub_sample']
    # RNNのタイプ(LSTM or GRU)
    enc_rnn_type = config['enc_rnn_type']
    # 中間層の次元数
    enc_hidden_dim = config['enc_hidden_dim']
    # Projection層の次元数
    enc_projection_dim = config['enc_projection_dim']
    # bidirectional を用いるか(Trueなら用いる)
    enc_bidirectional = config['enc_bidirectional']

    # Attention, Decoderの設定
    # RNN層のレイヤー数
    dec_num_layers = config['dec_num_layers']
    # RNN層の次元数
    dec_hidden_dim = config['dec_hidden_dim']
    # Attentionの次元
    att_hidden_dim = config['att_hidden_dim']
    # LocationAwareAttentionにおけるフィルタサイズ
    att_filter_size = config['att_filter_size']
    # LocationAwareAttentionにおけるフィルタ数
    att_filter_num = config['att_filter_num']
    # LocationAwareAttentionにおけるtemperature
    att_temperature = config['att_temperature']

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(output_dir, exist_ok=True)

    # 特徴量の平均/標準偏差ファイルを読み込む
    with open(mean_std_file, mode='r') as f:
        # 全行読み込み
        lines = f.readlines()
        # 1行目(0始まり)が平均値ベクトル(mean)，
        # 3行目が標準偏差ベクトル(std)
        mean_line = lines[1]
        std_line = lines[3]
        # スペース区切りのリストに変換
        feat_mean = mean_line.split()
        feat_std = std_line.split()
        # numpy arrayに変換
        feat_mean = np.array(feat_mean, 
                                dtype=np.float32)
        feat_std = np.array(feat_std, 
                               dtype=np.float32)

    # 次元数の情報を得る
    feat_dim = np.size(feat_mean)

    # トークンリストをdictionary型で読み込む
    # このとき，0番目は blank と定義する
    # (ただし，このプログラムではblankは使われない)
    token_list = {0: '<blank>'}
    with open(token_list_path, mode='r') as f:
        # 1行ずつ読み込む
        for line in f: 
            # 読み込んだ行をスペースで区切り，
            # リスト型の変数にする
            parts = line.split()
            # 0番目の要素がトークン，1番目の要素がID
            token_list[int(parts[1])] = parts[0]

    # <eos>トークンをユニットリストの末尾に追加
    eos_id = len(token_list)
    token_list[eos_id] = '<eos>'
    # 本プログラムでは、<sos>と<eos>を
    # 同じトークンとして扱う
    sos_id = eos_id

    # トークン数(blankを含む)
    num_tokens = len(token_list)
    
    # ニューラルネットワークモデルを作成する
    # 入力の次元数は特徴量の次元数，
    # 出力の次元数はトークン数となる
    model = MyMTLModel(dim_in=feat_dim,
                       dim_enc_hid=enc_hidden_dim,
                       dim_enc_proj=enc_projection_dim, 
                       dim_dec_hid=dec_hidden_dim,
                       dim_out=num_tokens,
                       dim_att=att_hidden_dim, 
                       att_filter_size=att_filter_size,
                       att_filter_num=att_filter_num,
                       sos_id=sos_id, 
                       att_temperature=att_temperature,
                       enc_num_layers=enc_num_layers,
                       dec_num_layers=dec_num_layers,
                       enc_bidirectional=enc_bidirectional,
                       enc_sub_sample=enc_sub_sample, 
                       enc_rnn_type=enc_rnn_type)
    
    # モデルのパラメータを読み込む
    model.load_state_dict(torch.load(model_file))

    # 訓練/開発データのデータセットを作成する
    test_dataset = SequenceDataset(feat_scp_test,
                                   label_test,
                                   feat_mean,
                                   feat_std)

    # 評価データのDataLoaderを呼び出す
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4)

    # CUDAが使える場合はモデルパラメータをGPUに，
    # そうでなければCPUに配置する
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    # モデルを評価モードに設定する
    model.eval()

    # デコード結果および正解ラベルをファイルに書き込みながら
    # 以下の処理を行う
    with open(hypothesis_file, mode='w') as hyp_file, \
         open(reference_file, mode='w') as ref_file:
        # 評価データのDataLoaderから1ミニバッチ
        # ずつ取り出して処理する．
        # これを全ミニバッチ処理が終わるまで繰り返す．
        # ミニバッチに含まれるデータは，
        # 音声特徴量，ラベル，フレーム数，
        # ラベル長，発話ID
        for (features, labels, feat_lens,
             label_lens, utt_ids) in test_loader:

            # PackedSequence の仕様上，
            # ミニバッチがフレーム長の降順で
            # ソートされている必要があるため，
            # ソートを実行する
            sorted_lens, indices = \
                torch.sort(feat_lens.view(-1),
                           dim=0,
                           descending=True)
            features = features[indices]
            feat_lens = sorted_lens

            # CUDAが使える場合はデータをGPUに，
            # そうでなければCPUに配置する
            features = features.to(device)

            # モデルの出力を計算(フォワード処理)
            # enc_lensは処理後のctc_outのフレーム数
            outputs, ctc_out, enc_lens = model(features,
                                               feat_lens)

            # バッチ内の1発話ごとに以下の処理を行う
            for n in range(outputs.size(0)):
                # 出力はフレーム長でソートされている
                # 元のデータ並びに戻すため，
                # 対応する要素番号を取得する
                idx = torch.nonzero(indices==n, 
                                    as_tuple=False).view(-1)[0]

                # 各ステップのデコーダ出力を得る
                _, hyp_per_step = torch.max(outputs[idx], 1)
                # numpy.array型に変換
                hyp_per_step = hyp_per_step.cpu().numpy()
                # 認識結果の文字列を取得
                hypothesis = []
                for m in hyp_per_step[:enc_lens[idx]]:
                    if m == eos_id:
                        break
                    hypothesis.append(token_list[m])

                # 正解の文字列を取得
                reference = []
                for m in labels[n][:label_lens[n]].cpu().numpy():
                    reference.append(token_list[m])
             
                # 結果を書き込む
                # (' '.join() は，リスト形式のデータを
                # スペース区切りで文字列に変換している)
                hyp_file.write('%s %s\n' \
                    % (utt_ids[n], ' '.join(hypothesis)))
                ref_file.write('%s %s\n' \
                    % (utt_ids[n], ' '.join(reference)))

