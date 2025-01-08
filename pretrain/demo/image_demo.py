# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser

from engine.mmengine.fileio import dump
from rich import print_json

from pretrain.mmpretrain.apis import ImageClassificationInferencer


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default = r"C:\Users\Bengi\Downloads\pose.jpg", help='Image file')
    parser.add_argument('--model', default=r"C:\Users\Bengi\PycharmProjects\sapiens\pretrain\configs\_base_\models\mae_vit-base-p16.py", help='Model name or config file path')
    parser.add_argument('--checkpoint', default= r"C:\Users\Bengi\Downloads\sapiens_0.3b_epoch_1600_clean.pth", help='Checkpoint file path.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Whether to show the prediction result in a window.')
    parser.add_argument(
        '--show-dir',
        type=str,
        help='The directory to save the visualization image.')
    parser.add_argument('--device', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    try:
        pretrained = args.checkpoint or True
        inferencer = ImageClassificationInferencer(
            args.model, pretrained=pretrained)
    except ValueError:
        raise ValueError(
            f'Unavailable model "{args.model}", you can specify find a model '
            'name or a config file or find a model name from '
            'https://mmpretrain.readthedocs.io/en/latest/modelzoo_statistics.html#all-checkpoints'  # noqa: E501
        )
    result = inferencer(args.img, show=args.show, show_dir=args.show_dir)[0]
    # show the results
    result.pop('pred_scores')  # pred_scores is too verbose for a demo.
    print_json(dump(result, file_format='json', indent=4))


if __name__ == '__main__':
    main()
