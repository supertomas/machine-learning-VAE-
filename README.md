# 2022年度の修士論文における「変分オートエンコーダーと祖先配列推定どちらが生物実験値を予測できるか」という題目における現状の研究結果<br>

## 1.2018年に発表した変分オートエンコーダーを扱った機械学習モデルと2021年に発表した機械学習モデルのどちらかが特定の配列の特徴を正確に捉えているのかを研究した。大腸菌配列とその1変異体のELBOをそれぞれのツールで算出した。そして各配列に対応する生物実験値とそれぞれのツールに対するELBOをプロットした。その結果を図2に示す。<br>
![2018年のコードと2021年のコードの性能](生物実験_Elbo_2018_2021.png "図2:祖先配列とその1変異体の特徴量と確率尤度の相関関係" )<br>
### 図2の左の結果から、生物実験値(反応速度を速める酵素活性値など)と現存生物の特性を表す機械学習モデルの出力値(ELBO)は正の相関があることがわかった。このことから、生物実験値が高い大腸菌の変異体は、現存生物の特徴を多く持った配列(言い換える野生型らしい配列)であることが言える。また、両者のグラフを比較すると2018年の方が生物実験値と出力値の相関が良い。よって2018年のモデルの方が現存生物の特徴を正確に捉えていることが言える。従って必ずしも最新の機械学習モデルが良い性能を示すとは限らないということがわかった。今後本当に2018年のモデルの方が優れているのかを確かめる予定である。

## 2.2018年に発表した機械学習モデルを使って初期生命の配列である祖先配列の現存生物の特徴を表した値を算出した。さらに、このリポジトリのものとは別のツールである祖先配列推定を使って、祖先最尤配列と1変異体の存在確率を示す確率尤度も算出した。<br>
### これら二つの値の相関グラフの結果を以下に示す。<br>
### 生成モデルである変分オートエンコーダー(VAE)は、アミノ酸配列の複数の残基の相互作用を読み取り特徴を出力できる特性がある。我々はこのVAEによって出力された入力したアミノ酸配列における特徴量値をELBOと呼ぶ。一方、祖先配列推定は系統樹情報を読み取ることで複数の生物種の類縁関係を読み取る特性がある。この特性を活かして算出したのが、その配列の存在確率を示す確率尤度である。
![祖先最尤配列とその1変異体のELBOと確率尤度の関係](祖先配列_確率尤度_ELBO.png "図1:祖先配列とその1変異体の特徴量と確率尤度の相関関係" )<br>
図1の結果から、祖先最尤配列と1変異体におけるELBOと確率尤度には概ね正の相関があることがわかった。このことから両者のツールにおける見ている情報は違えどそれぞれの配列の特性を捉えられているのが分かった。また、配列によってはELBOが高くなりやすいものもあれば、低くなりやすいものもあるということがわかった。


