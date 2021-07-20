# -*- coding: utf-8 -*-

#
# DNNを学習します．
#

# Pytorchを用いた処理に必要なモジュールをインポート
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

# 作成したDatasetクラスをインポート
from my_dataset import SequenceDataset

# 数値演算用モジュール(numpy)をインポート
import numpy as np

# プロット用モジュール(matplotlib)をインポート
import matplotlib.pyplot as plt

# hmmfunc.pyからMonoPhoneHMMクラスをインポート
from hmmfunc import MonoPhoneHMM

# モデルの定義をインポート
from my_model import MyDNN

# json形式の入出力を行うモジュールをインポート
import json

# os, sys, shutilモジュールをインポート
import os
import sys
import shutil

#
# メイン関数
#
if __name__ == "__main__":
    
    #
    # 設定ここから
    #

    # 訓練データの特徴量リスト
    train_feat_scp = \
        '../01compute_features/mfcc/train_small/feats.scp'
    # 訓練データのラベル(アライメント)ファイル
    train_label_file = \
        './exp/data/train_small/alignment'
    
    # 訓練データから計算された
    # 特徴量の平均/標準偏差ファイル
    mean_std_file = \
        '../01compute_features/mfcc/train_small/mean_std.txt'

    # 開発データの特徴量リスト
    dev_feat_scp = \
        '../01compute_features/mfcc/dev/feats.scp'
    # 開発データのラベル(アライメント)ファイル
    dev_label_file = \
        './exp/data/dev/alignment'

    # HMMファイル
    # HMMファイルは音素数と状態数の
    # 情報を得るためだけに使う
    hmm_file = '../03gmm_hmm/exp/model_3state_2mix/10.hmm'

    # 学習結果を出力するディレクトリ
    output_dir = os.path.join('exp', 'model_dnn')

    # ミニバッチに含める発話数
    batch_size = 5

    # 最大エポック数
    max_num_epoch = 60

    # 中間層のレイヤー数
    num_layers = 4

    # 中間層の次元数
    hidden_dim = 1024

    # splice: 前後 n フレームの特徴量を結合する
    # 次元数は(splice*2+1)倍になる
    splice = 5

    # 初期学習率
    initial_learning_rate = 0.008

    # 学習率の減衰やEarly stoppingの
    # 判定を開始するエポック数
    # (= 最低限このエポックまではどれだけ
    # validation結果が悪くても学習を続ける)
    lr_decay_start_epoch = 7

    # 学習率を減衰する割合
    # (減衰後学習率 <- 現在の学習率*lr_decay_factor)
    # 1.0以上なら，減衰させない
    lr_decay_factor = 0.5

    # Early stoppingの閾値
    # 最低損失値を更新しない場合が
    # 何エポック続けば学習を打ち切るか
    early_stop_threshold = 3

    #
    # 設定ここまで
    #

    # 出力ディレクトリが存在しない場合は作成する
    os.makedirs(output_dir, exist_ok=True)

    # 設定を辞書形式にする
    config = {'num_layers': num_layers, 
              'hidden_dim': hidden_dim,
              'splice': splice,
              'batch_size': batch_size,
              'max_num_epoch': max_num_epoch,
              'initial_learning_rate': initial_learning_rate,
              'lr_decay_start_epoch': lr_decay_start_epoch, 
              'lr_decay_factor': lr_decay_factor,
              'early_stop_threshold': early_stop_threshold}

    # 設定をJSON形式で保存する
    conf_file = os.path.join(output_dir, 'config.json')
    with open(conf_file, mode='w') as f:
        json.dump(config, f, indent=4)

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
    # 平均/標準偏差ファイルをコピーする
    shutil.copyfile(mean_std_file,
                    os.path.join(output_dir, 'mean_std.txt'))

    # 次元数の情報を得る
    feat_dim = np.size(feat_mean)

    # DNNの出力層の次元数を得るために，
    # HMMの音素数と状態数を得る
    # MonoPhoneHMMクラスを呼び出す
    hmm = MonoPhoneHMM()
    # HMMを読み込む
    hmm.load_hmm(hmm_file)
    # DNNの出力層の次元数は音素数x状態数
    dim_out = hmm.num_phones * hmm.num_states
    # バッチデータ作成の際にラベルを埋める値
    # はdim_out以上の値にする
    pad_index = dim_out
    
    # ニューラルネットワークモデルを作成する
    # 入力特徴量の次元数は
    # feat_dim * (2*splice+1)
    dim_in = feat_dim * (2*splice+1)
    model = MyDNN(dim_in=dim_in,
                  dim_hidden=hidden_dim,
                  dim_out=dim_out, 
                  num_layers=num_layers)
    print(model)

    # オプティマイザを定義
    # ここでは momentum stochastic gradient descent
    # を使用
    optimizer = optim.SGD(model.parameters(), 
                          lr=initial_learning_rate,
                          momentum=0.99)

    # 訓練データのデータセットを作成する
    # padding_indexはdim_out以上の値に設定する
    train_dataset = SequenceDataset(train_feat_scp,
                                    train_label_file,
                                    feat_mean,
                                    feat_std,
                                    pad_index,
                                    splice)
    # 開発データのデータセットを作成する
    dev_dataset = SequenceDataset(dev_feat_scp,
                                  dev_label_file,
                                  feat_mean,
                                  feat_std,
                                  pad_index,
                                  splice)
    
    # 訓練データのDataLoaderを呼び出す
    # 訓練データはシャッフルして用いる
    #  (num_workerは大きい程処理が速くなりますが，
    #   PCに負担が出ます．PCのスペックに応じて
    #   設定してください)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    # 開発データのDataLoaderを呼び出す
    # 開発データはデータはシャッフルしない
    dev_loader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)

    # クロスエントロピーを損失関数として用いる
    criterion = \
        nn.CrossEntropyLoss(ignore_index=pad_index)

    # CUDAが使える場合はモデルパラメータをGPUに，
    # そうでなければCPUに配置する
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)

    # モデルをトレーニングモードに設定する
    model.train()

    # 訓練データの処理と開発データの処理を
    # for でシンプルに記述するために，辞書データ化しておく
    dataset_loader = {'train': train_loader,
                      'validation': dev_loader}

    # 各エポックにおける損失値と誤り率の履歴
    loss_history = {'train': [],
                    'validation': []}
    error_history = {'train': [],
                     'validation': []}
    
    # 本プログラムでは，validation時の損失値が
    # 最も低かったモデルを保存する．
    # そのため，最も低い損失値，
    # そのときのモデルとエポック数を記憶しておく
    best_loss = -1
    best_model = None
    best_epoch = 0
    # Early stoppingフラグ．Trueになると学習を打ち切る
    early_stop_flag = False
    # Early stopping判定用(損失値の最低値が
    # 更新されないエポックが何回続いているか)のカウンタ
    counter_for_early_stop = 0

    # ログファイルの準備
    log_file = open(os.path.join(output_dir,
                                 'log.txt'),
                                 mode='w')
    log_file.write('epoch\ttrain loss\t'\
                   'train err\tvalid loss\tvalid err')

    # エポックの数だけループ
    for epoch in range(max_num_epoch):
        # early stopフラグが立っている場合は，
        # 学習を打ち切る
        if early_stop_flag:
            print('    Early stopping.'\
                  ' (early_stop_threshold = %d)' \
                  % (early_stop_threshold))
            log_file.write('\n    Early stopping.'\
                           ' (early_stop_threshold = %d)' \
                           % (early_stop_threshold))
            break

        # エポック数を表示
        print('epoch %d/%d:' % (epoch+1, max_num_epoch))
        log_file.write('\n%d\t' % (epoch+1))

        # trainフェーズとvalidationフェーズを交互に実施する
        for phase in ['train', 'validation']:
            # このエポックにおける累積損失値と発話数
            total_loss = 0
            total_utt = 0
            # このエポックにおける累積認識誤り文字数と総文字数
            total_error = 0
            total_frames = 0

            # 各フェーズのDataLoaderから1ミニバッチ
            # ずつ取り出して処理する．
            # これを全ミニバッチ処理が終わるまで繰り返す．
            # ミニバッチに含まれるデータは，
            # 音声特徴量，ラベル，フレーム数，
            # ラベル長，発話ID
            for (features, labels, feat_len,
                 label_len, utt_ids) \
                    in dataset_loader[phase]:

                # CUDAが使える場合はデータをGPUに，
                # そうでなければCPUに配置する
                features, labels = \
                    features.to(device), labels.to(device)

                # 勾配をリセット
                optimizer.zero_grad()

                # モデルの出力を計算(フォワード処理)
                outputs = model(features)

                # この時点でoutputsは
                # [バッチサイズ, フレーム数, ラベル数]
                # の3次元テンソル．
                # CrossEntropyLossを使うためには
                # [サンプル数, ラベル数]の2次元テンソル
                # にする必要があるので，viewを使って
                # 変形する
                b_size, f_size, _ =  outputs.size()
                outputs = outputs.view(b_size * f_size,
                                       dim_out)
                # labelsは[バッチサイズ, フレーム]の
                # 2次元テンソル．
                # CrossEntropyLossを使うためには
                # [サンプル数]の1次元テンソルにする
                # 必要があるので．viewを使って変形する．
                # 1次元への変形はview(-1)で良い．
                # (view(b_size*f_size)でも良い)
                labels = labels.view(-1)
                
                # 損失値を計算する．
                loss = criterion(outputs, labels)
                
                # 訓練フェーズの場合は，
                # 誤差逆伝搬を実行し，
                # モデルパラメータを更新する
                if phase == 'train':
                    # 勾配を計算する
                    loss.backward()
                    # オプティマイザにより，
                    # パラメータを更新する
                    optimizer.step()

                # 損失値を累積する
                total_loss += loss.item()
                # 処理した発話数をカウントする
                total_utt += b_size

                #
                # フレーム単位の誤り率を計算する
                #
                # 推定ラベルを得る
                _, hyp = torch.max(outputs, 1)
                # ラベルにpad_indexを埋めた
                # フレームを取り除く
                hyp = hyp[labels != pad_index]
                ref = labels[labels != pad_index]
                # 推定ラベルと正解ラベルが不一致な
                # フレーム数を得る
                error = (hyp != ref).sum()

                # 誤りフレーム数を累積する
                total_error += error
                # 総フレーム数を累積する
                total_frames += len(ref)
            
            #
            # このフェーズにおいて，1エポック終了
            # 損失値，認識エラー率，モデルの保存等を行う
            # 

            # 損失値の累積値を，処理した発話数で割る
            epoch_loss = total_loss / total_utt
            # 画面とログファイルに出力する
            print('    %s loss: %f' \
                  % (phase, epoch_loss))
            log_file.write('%.6f\t' % (epoch_loss))
            # 履歴に加える
            loss_history[phase].append(epoch_loss)

            # 総誤りフレーム数を，総フレーム数で
            # 割ってエラー率に換算
            epoch_error = 100.0 * total_error \
                        / total_frames
            # 画面とログファイルに出力する
            print('    %s error rate: %f %%' \
                  % (phase, epoch_error))
            log_file.write('%.6f\t' % (epoch_error))
            # 履歴に加える
            error_history[phase].append(epoch_error.cpu())

            #
            # validationフェーズ特有の処理
            #
            if phase == 'validation':
                if epoch == 0 or best_loss > epoch_loss:
                    # 損失値が最低値を更新した場合は，
                    # その時のモデルを保存する
                    best_loss = epoch_loss
                    torch.save(model.state_dict(),
                               output_dir+'/best_model.pt')
                    best_epoch = epoch
                    # Early stopping判定用の
                    # カウンタをリセットする
                    counter_for_early_stop = 0
                else:
                    # 最低値を更新しておらず，
                    if epoch+1 >= lr_decay_start_epoch:
                        # かつlr_decay_start_epoch以上の
                        # エポックに達している場合
                        if counter_for_early_stop+1 \
                               >= early_stop_threshold:
                            # 更新していないエポックが，
                            # 閾値回数以上続いている場合，
                            # Early stopping フラグを立てる
                            early_stop_flag = True
                        else:
                            # Early stopping条件に
                            # 達していない場合は
                            # 学習率を減衰させて学習続行
                            if lr_decay_factor < 1.0:
                                for i, param_group \
                                      in enumerate(\
                                      optimizer.param_groups):
                                    if i == 0:
                                        lr = param_group['lr']
                                        dlr = lr_decay_factor \
                                            * lr
                                        print('    (Decay '\
                                          'learning rate:'\
                                          ' %f -> %f)' \
                                          % (lr, dlr))
                                        log_file.write(\
                                          '(Decay learning'\
                                          ' rate: %f -> %f)'\
                                           % (lr, dlr))
                                    param_group['lr'] = dlr
                            # Early stopping判定用の
                            # カウンタを増やす
                            counter_for_early_stop += 1
                    
    #
    # 全エポック終了
    # 学習済みモデルの保存とログの書き込みを行う
    #
    print('---------------Summary'\
          '------------------')
    log_file.write('\n---------------Summary'\
                   '------------------\n')

    # 最終エポックのモデルを保存する
    torch.save(model.state_dict(), 
               os.path.join(output_dir,'final_model.pt'))
    print('Final epoch model -> %s/final_model.pt' \
          % (output_dir))
    log_file.write('Final epoch model ->'\
                   ' %s/final_model.pt\n' \
                   % (output_dir))

    # 最終エポックの情報
    for phase in ['train', 'validation']:
        # 最終エポックの損失値を出力
        print('    %s loss: %f' \
              % (phase, loss_history[phase][-1]))
        log_file.write('    %s loss: %f\n' \
                       % (phase, loss_history[phase][-1]))
        # 最終エポックのエラー率を出力    
        print('    %s error rate: %f %%' \
              % (phase, error_history[phase][-1]))
        log_file.write('    %s error rate: %f %%\n' \
                       % (phase, error_history[phase][-1]))

    # ベストエポックの情報
    # (validationの損失が最小だったエポック)
    print('Best epoch model (%d-th epoch)'\
          ' -> %s/best_model.pt' \
          % (best_epoch+1, output_dir))
    log_file.write('Best epoch model (%d-th epoch)'\
          ' -> %s/best_model.pt\n' \
          % (best_epoch+1, output_dir))
    for phase in ['train', 'validation']:
        # ベストエポックの損失値を出力
        print('    %s loss: %f' \
              % (phase, loss_history[phase][best_epoch]))
        log_file.write('    %s loss: %f\n' \
              % (phase, loss_history[phase][best_epoch]))
        # ベストエポックのエラー率を出力
        print('    %s error rate: %f %%' \
              % (phase, error_history[phase][best_epoch]))
        log_file.write('    %s error rate: %f %%\n' \
            % (phase, error_history[phase][best_epoch]))

    # 損失値の履歴(Learning Curve)グラフにして保存する
    fig1 = plt.figure()
    for phase in ['train', 'validation']:
        plt.plot(loss_history[phase],
                 label=phase+' loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig1.legend()
    fig1.savefig(output_dir+'/loss.png')

    # 認識誤り率の履歴グラフにして保存する
    fig2 = plt.figure()
    for phase in ['train', 'validation']:
        plt.plot(error_history[phase],
                 label=phase+' error')
    plt.xlabel('Epoch')
    plt.ylabel('Error [%]')
    fig2.legend()
    fig2.savefig(output_dir+'/error.png')

    # ログファイルを閉じる
    log_file.close()

