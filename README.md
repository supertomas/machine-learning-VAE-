# 2022年度の修士論文における「機械学習と祖先配列推定を活性が高い祖先配列をデザイン」という題目における研究概要と現状の研究結果<br>

## 研究概要：<br>
### 現代の生物種は全て図1のように共通祖先から進化を遂げて枝分かれしていき、現代の生物種に至ったと言われている。この共通祖先の配列を我々祖先配列と呼ぶ。この配列は現代の生物種のアミノ酸配列をコンピュータツールに入力すれば複数の候補配列が手に入る。そのツールを祖先配列推定と呼び、生物種の類縁関係の情報から祖先配列の部位ごとのアミノ酸を推定することができ複数の祖先候補配列を得ることができる。しかし、本来アミノ酸配列で構成されるタンパク質というのは折りたたまって立体構造の形をとる。立体構造によって配列上では離れていた部分が互いに相互作用を及ぼすことによって特有の機能を発揮する。祖先配列推定はこの配列の部位ごとの相互作用を考慮していない。そこで扱われるのが機械学習である。機械学習の中でも変分オートエンコーダーという生成モデルは残基間の相互作用を読み取り調べたい配列が学習データとどれほど合致しているかを数値として出力することができる。この祖先配列推定と機械学習モデルの二つを組み合わせれば存在確率が高く機能が高い祖先配列を推定することができ、その配列を生物実験班に提供することができる。その配列を温度やphなど様々な条件下で試し、太古の生物はどのような環境下で生きていたかを探求することが本研究の目的である。
![2018年のコードと2021年のコードの性能](系統樹.png "図1:系統樹" )<br>


## 研究結果1<br>
### 2018年に発表した変分オートエンコーダーを扱った機械学習モデルと2021年に発表した機械学習モデルのどちらかが特定の配列の特徴を正確に捉えているのかを研究した。大腸菌配列とその1変異体のELBOをそれぞれのツールで算出した。そして各配列に対応する生物実験値とそれぞれのツールに対するELBOをプロットした。その結果を図2に示す。<br>
![2018年のコードと2021年のコードの性能](生物実験_Elbo_2018_2021比較.png "図2:祖先配列とその1変異体の特徴量と確率尤度の相関関係" )<br>
### 図2の左の結果から、生物実験値(反応速度を速める酵素活性値など)と現存生物の特性を表す機械学習モデルの出力値(ELBO)は正の相関があることがわかった。このことから、生物実験値が高い大腸菌の変異体は、現存生物の特徴を多く持った配列(言い換える野生型らしい配列)であることが言える。また、両者のグラフを比較すると2018年の方が生物実験値と出力値の相関が良い。よって2018年のモデルの方が現存生物の特徴を正確に捉えていることが言える。しかし、これはあくまで先行研究が扱ったパラメータのままで算出した結果である。そのためこの結果で使われたパラメータが最も精度良く配列の相互作用を捉えているとは限らない。そのため、先行研究で使われていなかった他のパラメータも試し、パラメータが相互作用を捉える精度に影響が出るのかを今後試す予定である。その結果から本当に2018年のモデルの方が優れているのかを確かめる予定である。yht

## 研究結果2
### 2018年に発表した機械学習モデルを使って初期生命の配列である祖先配列の現存生物の特徴を表した値を算出した。さらに、このリポジトリのものとは別のツールである祖先配列推定を使って、祖先最尤配列と1変異体の存在確率を示す確率尤度も算出した。<br>
### これら二つの値の相関グラフの結果を以下に示す。<br>
![祖先最尤配列とその1変異体のELBOと確率尤度の関係](祖先配列_確率尤度_ELBO.png "図3:祖先配列とその1変異体の特徴量と確率尤度の相関関係" )<br>
### 図3の結果から、祖先最尤配列と1変異体におけるELBOと確率尤度には概ね正の相関があることがわかったため、より存在確率が高く機能の高い祖先配列を見つけられる期待が高まった。またこの結果から、存在確率が低いけど、共通性質の値が高い配列やその逆のパターンも存在することがわかった。このことから、最尤配列から変異を加えたことで新たな相互作用が働き、活性が上がった配列があることが言える。そこで今度は2変異体のelboを調べることでどの残基どうしのどのアミノ酸が強い相互作用が働き活性があがる傾向にあるのかを調べる。


