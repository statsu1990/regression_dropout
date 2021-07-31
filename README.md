# 回帰問題でのDropoutの影響検証
## 詳細
https://github.com/statsu1990/regression_dropout/blob/main/regression_dropout.ipynb

## 背景・目的
回帰問題をNNで解くときにDropoutとBatchNormを使わない方が良いという話を聞いた。  
https://twitter.com/Tsultra_K/status/1421057421527506944  
https://towardsdatascience.com/pitfalls-with-dropout-and-batchnorm-in-regression-problems-39e02ce08e4d  

回帰問題でDropoutを使うことによる影響を検証し、その対策について検討する。  
(BatchNormの検証はしていない。)  

## まとめ
- Dropoutによる悪影響
  - 回帰問題でDropoutを使うと、正解値の絶対値が大きいときに推定値が小さくなる傾向があった。Dropoutの後に非線形層(ReLUなど)があるとその効果は特に顕著であった。
- 対策
  - Dropoutの代わりにGaussianDropout, UniformDropoutを使うと悪影響がある程度改善された。
  - MontecarloDropoutを使うと悪影響がなくなった。
- 考察
  - 対策手法ではDropout適用後の分散の変化が小さくなる(と思う)ので、対策により結果が改善されたと考えられる。
- その他
  - 分類問題ではDropoutを使っても悪影響はなかった。

## Dropoutについて
以下ではh = Dropout(x)とする。
- Dropoutは学習中に一定の確率でxの値を0にする手法。推論時は0にしない。過学習を抑制するために使われる。
- ランダムにxの値が0になるため、Dropoutを通す場合(学習時)と通さない場合(推論時)ではhの統計量が変わる。学習と推論でhの平均値が変わらないような工夫がされているが、分散は変わってしまう。
- hの統計量が変わるため、Dropout以降の層で学習時と推論時で不整合が起きる。これが性能に悪影響を与えることがある。  
  (BNとDropoutを同時に使うと性能が落ちることがあるのもこれが原因らしい。)
- Dropoutの亜種には以下のものがある。
  - GaussianDropout: h = x + x * ε (εはガウスノイズ)
  - UniformDropout: h = x + x * ε (εは一様分布)
  - AlphaDropout: xが平均0、標準偏差1であればAlphaDropout(x)も平均0、標準偏差1になるらしい？SELUと一緒に使うといいらしい。ちゃんと理解していない。  
  - MontecorloDropout: 学習時と同じようにDropoutを有効にした状態で複数回推論し、その平均値を推定値とする。

## 検証条件・内容
- 1次元座標上に設置された長方形の面積を1D CNNで推定する回帰タスクを検証対象とした。
- 学習データとテストデータは同じものとした。これにより、学習とテストでデータが同じという理想条件でどのような問題が起きるか確認できる。  
  また、このような条件であるため汎化性能については検討できない。
- Dropoutの有無、Dropoutの位置、drop rateの大きさ、Dropout手法を変えて影響を確認した。

## 検証結果
- Dropoutを使わなければ学習とテストの誤差は一致し、推定値と正解値の誤差はほぼ0になった。(学習データとテストデータが同じである & 過学習する ので当然)
- Dropoutを使った場合、以下の問題が発生した。
  - 正解値が大きいときほどテストデータの誤差が大きくなった(推定値<正解値という傾向)。Dropoutの後ろに非線形層(ReLUなど)があるときほどこの効果は顕著であった。学習epochが増えてもある程度誤差がある状態で収束した。  
    学習データではこの問題は発生しなかった。
- 上記の問題への対策とその効果は以下のとおり。
  - GaussianDropout: 普通のDropoutよりはマシだが、問題解消せず。
  - UniformDropout: 普通のDropout、GaussianDropoutよりはだいぶマシだが、問題解消せず。
  - AlphaDropout: 普通のDropoutよりはマシっぽいがよくわからない。問題解消せず。AlphaDropoutだけは推定値>正解値という傾向。(SELUと一緒に使うものらしいが、面倒なのでSELU使っていない)
  - MontecarloDropout: 問題解消した。
  - 期待値計算(CE使わないAttention like): sum(Attention * softmax(value))で期待値を計算するように回帰した。問題解消せず。シードによって挙動が結構変わるので不安定っぽい。

## 検証結果(おまけ)
- 分類問題ではDropoutを使っても問題発生しなかった。

## 考察
- GaussianDropout、UniformDropout、MontecarloDropoutではDropout適用後の分散の変化が小さくなる(と思う)ので、対策により結果が改善されたと考えられる。

## 参考文献
- Pitfalls with Dropout and BatchNorm in regression problems  
  https://towardsdatascience.com/pitfalls-with-dropout-and-batchnorm-in-regression-problems-39e02ce08e4d
- 回帰でDropoutやBNを使ってはいけないという話を試してみた  
  https://github.com/ak110/regression-dropoutbn
- GaussianDropout  
  http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf  
  https://keras.io/ja/layers/noise/
- UniformDropout  
  https://arxiv.org/pdf/1801.05134.pdf
- Montecarlo Dropout  
  https://arxiv.org/pdf/1506.02142.pdf
