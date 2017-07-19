# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from sys import stderr

from .commands import discriminate, generate, train


def parse_arg():
    parser = ArgumentParser(description='Generative Adversarial Nets Utility.')
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train', help='GANを訓練する.')
    train_parser.add_argument('discriminator', type=str, help='discriminatorモデルアーキテクチャ.'
                                                              'サポートしているファイルフォーマットは[.json|.yml].'
                                                              'kerasの仕様に従ったものでなければならない.')
    train_parser.add_argument('generator', type=str, help='generatorモデルアーキテクチャ.'
                                                          'サポートしているファイルフォーマットは[.json|.yml].'
                                                          'kerasの仕様に従ったものでなければならない.')
    train_parser.add_argument('dinput', type=str, help='discriminatorの訓練に使用する入力データセットのファイル名.'
                                                       'サポートしているファイルフォーマットは[.npy].')
    train_parser.add_argument('ginput', type=str, help='generatorの訓練に使用する入力データセットのファイル名.'
                                                       'サポートしているファイルフォーマットは[.npy].')
    train_parser.add_argument('save', type=str, help='学習結果を保存するディレクトリ.'
                                                     '存在しない場合は終了する.')
    train_parser.add_argument('--epoch', default=20,
                              type=int, help='エポックサイズ.デフォルトは20回.')
    train_parser.add_argument('--batch', default=32,
                              type=int, help='バッチサイズ.デフォルトは32.')
    train_parser.add_argument('--dweight', type=str, help='discriminatorの学習済み重みパラメータ.'
                                                          '指定したdiscriminatorのアーキテクチャでなければならない.'
                                                          'サポートしているファイルフォーマットは[.h5].')
    train_parser.add_argument('--dlr', default=0.0002,
                              type=float, help='discriminatorの学習係数.')
    train_parser.add_argument('--dbeta1', default=0.5, type=float, help='discriminatorのAdam Optimizerのbeta1の値.'
                                                                        'デフォルトは0.5.')
    train_parser.add_argument('--gweight', type=str, help='generatorの学習済み重みパラメータ.'
                                                          '指定したgeneratorのアーキテクチャでなければならない.'
                                                          'サポートしているファイルフォーマットは[.h5].')
    train_parser.add_argument('--glr', default=0.0002,
                              type=float, help='generatorの学習係数.デフォルトは0.0002.')
    train_parser.add_argument('--gbeta1', default=0.5, type=float, help='generatorのAdam Optimizerのbeta1の値.'
                                                                        'デフォルトは0.5.')
    train_parser.set_defaults(func=train)

    discriminative_parser = subparsers.add_parser(
        'discriminate', help='学習済みのDiscriminatorモデルを用いて識別を行う.')
    discriminative_parser.add_argument('model', type=str, help='学習済みdiscriminatorモデル.'
                                                               'サポートしているファイルフォーマットは[.h5].'
                                                               'kerasの仕様に従ったものでなければならない.')
    discriminative_parser.add_argument('x', type=str, help='識別に使用されるデータセット.'
                                                           'サポートしているファイルフォーマットは[.npy].')
    discriminative_parser.add_argument('save', type=str, help='結果を保存するファイルパス.拡張子がない場合は[.npy]で保存される.'
                                                              'また,ディレクトリが存在しない場合は終了し,ファイルがすでに存在する場合は上書きする.')
    discriminative_parser.add_argument(
        '-b', '--batch', default=32, type=int, help='バッチサイズ.デフォルトは32.')
    discriminative_parser.set_defaults(func=discriminate)

    generative_parser = subparsers.add_parser(
        'generate', help='学習済みのGeneratorモデルを用いて生成を行う.')
    generative_parser.add_argument('model', type=str, help='学習済みgeneratorモデル.'
                                                           'サポートしているファイルフォーマットは[.h5].'
                                                           'kerasの仕様に従ったものでなければならない.')
    generative_parser.add_argument('x', type=str, help='生成に使用される入力データセット.'
                                                       'サポートしているファイルフォーマットは[.npy].')
    generative_parser.add_argument('save', type=str, help='結果を保存するファイルパス.拡張子がない場合は[.npy]で保存される.'
                                                          'また,ディレクトリが存在しない場合は終了し,ファイルがすでに存在する場合は上書きする.')
    generative_parser.add_argument(
        '-b', '--batch', default=32, type=int, help='バッチサイズ.デフォルトは32.')
    generative_parser.set_defaults(func=generate)
    return parser.parse_args()


def main():
    args = parse_arg()
    try:
        args.func()
    except Exception as e:
        print(e, file=stderr)
        exit(1)


if __name__ == '__main__':
    main()
