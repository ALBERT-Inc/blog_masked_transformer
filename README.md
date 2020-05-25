# blog_masked_transformer
MaskedTransformer再現実装公開用リポジトリ

## セットアップ
必要なライブラリと、実装されたMasked Transformerパッケージをセットアップします。なお、実行した環境はPython3.6で、CUDA 10.1です。
```
pip install .
python -m spacy download en
```
データセットは[[train (9.6GB)](http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/training_feat_yc2.tar.gz), [val (3.2GB)](http://youcook2.eecs.umich.edu/static/dat/yc2_densecap/validation_feat_yc2.tar.gz)]からダウンロードできます。これを適当なディレクトリ`/path/to/data`に配置して解凍してください。

公開されている学習済み重みファイルは[こちら](http://youcook2.eecs.umich.edu/static/dat/densecap_checkpoints/pre-trained-models.tar.gz)からダウンロードできます。これを解凍し、`yc2-2L-e2e-mask`というディレクトリの下にある重みファイルを利用して学習してください。

## 学習
以下のコマンドを実行することで学習が実行されます。
```
masked_transformer train config/train.yml --dataset_root /path/to/data --device cuda:0 --num_workers 8
```

## 評価
以下のコマンドを実行することでBLEUでの評価結果が出力されます。
```
masked_transformer evaluate config/eval.yml --dataset_root /path/to/data --device cuda:0 --num_workers 8
```