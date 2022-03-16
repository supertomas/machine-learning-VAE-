# 2018年に発表した変分オートエンコーダー(VAE)を使用してタンパク質の変異の影響や生物実験活性を予測する計算機ツール
## このコードは、2018年にAdam J.Riesselmanが開発したアミノ酸配列の座位ごとの相関を読み取ることでそれらの特徴を捉え調べたい配列の変異の影響を予測できるツールを改変したものである。このコードの論文に関するURLを以下に示す。<br>
https://www.nature.com/articles/s41592-018-0138-4<br>
## Python 2.7とTheano 1.0.1のバージョン化で走らせることができます。仮想環境下を作りGPUによる計算を行う場合は、CUDAを別途インストールする必要があります。