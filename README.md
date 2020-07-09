# GA_car_simulaton
利用 RBFN+GA 進行 car simulaton
### DataPoint.py
此份程式碼設計沿用作業一的資料結構，去掉 Fuzzy 用到幾個 class，並加定義 gene class 作為 data structurer 使用，一個 gene 紀錄的就是一組 rbf 的參數。每個 input vector 長度可能 3 或 5，故每個 gene 要記得的 mean 的數量是 input vector dim * rbf_units 
Deviation 部分，原本想順著上次作業的邏輯，將範圍界定在車子的直徑以內，但收斂速度太慢
了，所以有試圖調成 1 跟 10，10 不知道為什麼感覺上快很多，所以最後決定用 10。

### GA.py
基因演算法實作部分，分別定義 reproduction()、select_and_crossover()、adaptaive_func()、mutation() 等幾個大的 function 逐一實作，終止條件為 iteration 達指定次數。
GA_compute()為主要流程控制的 function，前端介面會呼叫這個 function 並傳入指定參數，並呼叫上述提到的幾個 function 執行基因演算法。其中在 select_and_crossover 與 adaptaive_func() 又各自拆成幾個小 function 進行實現。
1. 讀檔與初始化 population
input_dim, x_upper, x_down, traindata = read_training(file)
2. adaptaive_func()：計算每個 gene 的適應程度，因為是透過 rbf 訓練，故會呼叫 RBF() 進行計算，並回傳適應函數與均方差結果，另因作業提供的適應函數算法是越小越好，又後面輪盤法會用大適應函數的值，為方便計算，我有將函數倒數，這樣才可以保證越大的值其被選擇的機率要越高，原本有作正規化，但發現反正規化的值很奇怪，問同學後說可以直接乘以 40 縮放。

3. reproduction()：又分輪盤法與競爭法，輪盤法的 population 會根據自適應函數與平均值相除的大小，決定要複製的數量的機率，放入交配池中，當自適應函數越大，被選入的機率越高。競爭法則是先用隨機方式，選出 5 個 index，互相比較，將適應函數最大的放入交配池中，重複 N 次直到交配持數量等於使用者輸入的族群數量。

4. select_and_crossover()：作業採用實數型編碼方式，此 function 主要控制基因是否要交配，若交配機率大於指定的交配律則呼叫 crossover() 進行交配，交配公式參考講義：由於交配方式是採左右相鄰兩個基因進行交配，故需要區分 population 數量為奇數或偶數，因此程式會先進行判斷，其餘實作方式一樣，唯一差別在奇數的最後一個基因不會被交配到。

5. mutation()：此 function 進行突變，若突變機率大於輸入的機率，則基因發生突變，突變根據以下公式進行其中 s 為控制雜訊之大小，故每個 iteration 的突變，設定的 s 相同，為介於-1~1 之間的浮點數，又我以隨機的方式選取+1 或是-1，讓每次 noise 的突變式增加或是遞減變得更不固定，雖然跟公式不太相符，但我發現作這個步驟後，比較容易突然突變出好基因，讓模型error_rate 突然下降。random_noise 部分，因為邏輯上應該也是要 random 一個 gene 作為 noise，故我直接 new 一個 Gene 的物件作為 noise， 同時為了區別是 mutation 發生更新還是 crossover 發生更新，這個 function 若突變發生，則適應函數在改成 None，並將檢查的 flag 設定成 true。

### (3) 實驗心得：
我發現兩次 error 逼近到差不多的值（根據本作業算法大約為 9），才可以讓模型成功走到最後。6D 需要的參數 node 數與迭代、族群數量都要調大，而且非常久，才有可能找到比較好的參數，用跟 4D 一樣的參數，或是調整圖變率跟交配率，大部分的時候會卡在 9.4 就降不下去，但9.4 的 error_rate 會在第二個轉彎撞牆。原始參數中，在 4D 中，error_rate 印象降到約 9.2左右會行走成功。後來抱持著算了我就讓你慢慢迭代，期望有天圖便可以到好 case 的情況直接調大成兩倍的 itteration，並讓基因數更多一點，最後讓 6D 成功找到可以走到終點的參數，6D 實驗的成功 Case error_rate 約在 9.0，因此我猜測這個模型只要能收斂到 9.2 以下就可以成功過

本次作業的測資。另外相較於看 error_rate，我發現看適應函數會更準確，曾經有試過 case error 降到 8.~，但無法成功走到終點，印出適應函數檢查才發現他的適應函數其實低於error_rate 較高但可以正確走到終點的基因，因此本來預計要用 error_rate 標準作為回傳值，可以停掉 iteration 讓程式加速，最後還是改成讓 iteration 慢慢跑完。
