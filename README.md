Glasses Design System

このリポジトリでは、Python を用いたメガネのデザインおよび視覚化のスクリプトを提供します。glasses2.py はパラメータを指定して様々なメガネの形状を描画できるスクリプトです。

ファイル構成

glasses2.py: メガネのデザインをパラメトリックに生成するスクリプト。
glasses1（別ファイル）: 顔のパーツを変更可能。こちらはメガネのデザインだけでなく、顔の要素（目、鼻、口など）にも対応。
必要なライブラリ

matplotlib
numpy
インストール:

pip install matplotlib numpy
使用方法

以下は、glasses2.py を使ってメガネを描画する方法です。

from glasses2 import draw_glasses

draw_glasses(
    bridge_width=20,
    lens_width=50,
    lens_height=30,
    frame_thickness=5,
    temple_length=100,
    roundness=0.8
)
パラメータ一覧（glasses2.py）

パラメータ名	説明	型	例
bridge_width	メガネのブリッジ幅（中央の幅）	float	20
lens_width	レンズの横幅	float	50
lens_height	レンズの高さ	float	30
frame_thickness	フレームの太さ	float	5
temple_length	テンプル（つる）の長さ	float	100
roundness	レンズの丸み（0: 四角, 1: 丸）	float	0.0〜1.0
glasses1 との違い

glasses2.py: メガネのデザインのみに特化。設計パラメータによって形状を変更できます。
glasses1: 顔のパーツ（目、鼻、口など）を変更できる設計。メガネだけでなく、顔の全体的な表現が可能です。
