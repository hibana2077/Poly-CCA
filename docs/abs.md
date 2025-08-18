# 論文題目（可投 IEEE/ACM TCBB）

**Poly-CCA：對多項式擾動不變的多群典型相關表示，用於小樣本 DNA 分類**

（英文：*Poly-CCA: Polynomial-Perturbation-Invariant Multi-Group Canonical Representations for Small-Sample DNA Classification*）

---

# 研究核心問題

在 DNA 序列分類（如 promoter 與 splice junction）中，小樣本與結構化雜訊（替換、插入/刪除、測序平台特徵）會顯著影響以 k-mer 為基底的表示與分類器的穩定性。本研究欲回答：**能否在可控的「多項式生成＋數學可解析的雜訊」模型下，構造對此類擾動不敏感、且以 CPU 即可訓練/驗證的「典型相關（CCA）式」表示，並在真實小型 DNA 資料集上證明其效益？**

---

# 研究目標

1. **PolySimDNA 合成基準**：提出以「多項式 motif 產生器」生成二類 DNA 序列（含可控背景、GC 偏好），並提供**可解析的替換/插入/刪除雜訊公式**。
2. **Poly-CCA 表示學習**：把每條序列轉成 k-mer 向量 $x$，再做**低階多項式展開**（度數 $d=2$）得到 $\psi_d(x)$。以**兩視角/多群 CCA**在「原序列 vs. 擾動序列」、「不同雜訊強度群組」間對齊，學得對雜訊不變的子空間。
3. **理論保證**：給出在「獨立隨機替換模型」下，$\psi_d(\cdot)$ 之**期望收縮（shrinkage）係數**與**CCA 恢復共享訊號子空間**之條件。
4. **CPU 友善**：選 $k=3$（64 維），$d=2$（展開後維度約 2K 等級），全流程在**單機 CPU**即可完成。
5. **實證**：在 UCI **Promoter** 與 **Splice-junction** 小型資料集，以及 PolySimDNA 上驗證，與標準 k-mer + 線性/核方法比較。資料集與 k-mer 在生物資訊中的標準性與可重現性見近年的綜述。 ([archive.ics.uci.edu][1], [PMC][2])

---

# 貢獻與創新（偏數學理論；已先行可行性驗證）

1. **可解析的雜訊模型 → 對 k-mer/多項式特徵的封閉形推導**
   在獨立替換率 $p$ 下，**一個長度為 $k$ 的 k-mer 保持不變的機率為 $(1-p)^k$**；因此置中後的 k-mer 計數向量其**期望**呈**對角收縮**（每個座標乘上 $(1-p)^k$）＋一個與背景分布相關的常數偏移（置中後消失）。此結論與經典 k-mer 錯誤行為一致（單一鹼基錯誤會改變所有重疊的 k-mers）。我們將此推廣到**度 $d$ 的多項式單項式**，得到對應的收縮因子，形成**解析可解的雜訊傳播公式**。 ([BioMed Central][3])
2. **CCA 噪聲不變子空間的充要條件（兩視角）**
   以兩視角 $(X,Y)$ 表示「原始」與「擾動」特徵（$\psi_d$ 之後）。若雜訊為零均值、與訊號獨立，則 $\Sigma_{XY}=\Sigma_S$ 而自相關多出噪聲項，**CCA 的主特徵向量回收共享訊號子空間**（概率/白化版 pCCA 視角）。我們把上述**收縮對角化**代入一般化特徵值問題，證明**特徵向量不變、特徵值僅縮放**。 ([CiteSeerX][4], [PMC][5])
3. **多群 MG-TCCA 擴展（小樣本、異質雜訊）**
   將不同雜訊率、不同背景（甚至不同資料集）視為**群組**，用 **MG-TCCA** 的群組共享訊號視角整合，以解析式收縮矩陣作為先驗，提升穩健性與統計效率。 ([ACM Digital Library][6], [PMC][7])
4. **CPU 可重現與 TCBB 專刊對齊**
   方法完全由**線性代數＋低階多項式展開＋CCA**構成，配合小型資料集，可在單機 CPU 完成；研究方向契合 TCBB 對**演算法與數理方法**的重視。 ([Scimago][8])

---

# 數學理論推演與證明（精要版）

**符號與生成**

* 字母表 $\mathcal{A}=\{A,C,G,T\}$，序列長 $L$。
* k-mer 映射 $\phi_k:\text{seq}\to\mathbb{R}^{4^k}$ 取（正規化）k-mer 計數。多項式展開 $\psi_d(x)$ 包含至多度 $d$ 的單項式。
* 兩視角：$X=\psi_d(\phi_k(S))$，$Y=\psi_d(\phi_k(\tilde S))$，其中 $\tilde S$ 為 $S$ 經獨立替換率 $p$ 的擾動；置中處理（整體或類別內）。

**引理 1（k-mer 保留機率與期望收縮）**
在獨立替換模型下，任一特定 k-mer 在長度 $L$ 的序列中**保持不變**之機率為 $(1-p)^k$。因此置中後

$$
\mathbb{E}[\phi_k(\tilde S)\mid \phi_k(S)]=D_p\,\phi_k(S),\quad D_p=(1-p)^k I,
$$

而度 $d$ 的單項式（由 $m$ 個座標相乘）之期望縮放為 $(1-p)^{km}$。
*證意*：單一替換會影響所有覆蓋該位點的 k-mers（見 Quake 的錯誤-k-mer 關係），獨立性給出乘法形式。對應至單項式為座標乘積的線性組合，故縮放為指數相加。 ([BioMed Central][3])

**定理 1（CCA 子空間不變性）**
令 $X=\psi_d(\phi_k(S))$，$Y=\psi_d(\phi_k(\tilde S))$。假設：
(i) $S$ 的共享訊號子空間協方差為 $\Sigma_S\succ0$；
(ii) 擾動對 $\psi_d$ 的影響滿足引理 1 的對角收縮且噪聲零均值、與 $S$ 獨立；
(iii) 以總體置中或類別內置中消去常數偏差。
則

$$
\Sigma_{XY}=D_p\Sigma_S,\quad 
\Sigma_{XX}=D_p\Sigma_S D_p + \Sigma_{\varepsilon_X},\quad
\Sigma_{YY}=\Sigma_S + \Sigma_{\varepsilon_Y},
$$

CCA 的廣義特徵問題

$$
\Sigma_{XX}^{-1}\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX} \, u=\rho^2 u
$$

之**特徵向量與 $\Sigma_S$ 之主子空間一致**（至正交變換），$\rho$ 僅受 $D_p$ 與雜訊能量縮放。
*證意（綱要）*：pCCA/白化-CCA 視角下，若交叉協方差只含共享項，噪聲只進入自協方差，則在白化後的問題上，特徵向量等同於共享訊號的主方向。將引理 1 的 $D_p$ 代入可得。 ([CiteSeerX][4], [PMC][5])

**定理 2（多群 MG-TCCA 一致性直觀）**
將不同 $p$ 或不同背景/資料集視為群 $g=1,\dots,G$，每群對角收縮 $D_{p_g}$ 與噪聲協方差不同，但共享 $\Sigma_S$。MG-TCCA 目標最大化群間共同相關，等價於在多個白化座標系上尋找一致的共享子空間；在訊號分離度與群內雜訊有限條件下，可恢復共享子空間。
*註*：此與 MG-TCCA 於多子群共享訊號的理論脈絡一致。 ([ACM Digital Library][6], [PMC][7])

---

# 預計使用 dataset（皆為小型、可 CPU）

* **UCI Promoter Gene Sequences**：106 條 E. coli promoter/非 promoter，長度 57，經典小樣本。 ([archive.ics.uci.edu][1], [search.r-project.org][9])
* **UCI Splice-junction Gene Sequences**：3,190 條，三類（EI/IE/none），分類任務明確。 ([archive.ics.uci.edu][10])
* **PolySimDNA（本研究釋出）**：以多項式 motif 產生器生成的二類序列＋可控替換/插入/刪除雜訊，提供真值與噪聲參數。
* **k-mer 設定**：$k=3$（64 維），多項式度 $d=2$；此維度下 CCA/線性分類均可在單機 CPU 完成，並貼合 k-mer 方法在生物資訊領域的常見做法與新近觀察。 ([PMC][2])

---

# 與現有研究之區別

* **不同於「拓樸/連結預測」脈絡**：近年生醫網路的拓樸特徵或相似度能提升**連結預測**（如 PPI/組織特異網路）效能；本研究則聚焦**序列分類**，透過 CCA 找到跨雜訊條件的\*\*「典型（canonical）表示」\*\*，概念上與「為網路找 canonical representation」相呼應，但物件與任務皆不同。 ([PMC][11], [ResearchGate][12])
* **不同於微生物群落的共現網路微差分析**：該線路以**圖/網路層級**的特徵推論小樣本差異；我們在**序列層級**用解析雜訊模型＋CCA 對齊共享訊號，理論與實作更輕量。 ([arXiv][13])
* **不同於共識樹/演化樹演算法**：頻率差共識樹屬於**彙整多棵系統發生樹**的演算法設計；我們不建樹，而是做**表示學習＋線性代數**，資源需求小得多。 ([drops.dagstuhl.de][14])
* **理論上明確處理測序雜訊→k-mer 的效應**：我們給出**封閉形的收縮係數**與 CCA 子空間不變條件，連到 pCCA 理論；這在傳統 k-mer 文獻中多以經驗或模擬描述，較少與 CCA 做到解析結合。 ([BioMed Central][3], [CiteSeerX][4])
* **貼合 TCBB 的演算法/數理取向**：工作重心為**可證明的統計/線代方法**與可重現的**輕量實作**。 ([Scimago][8])

---

# 實作綱要（可直接落地）

* **前處理**：one-hot → $k=3$ k-mer 計數 → 標準化；
* **特徵**：二次多項式展開（可用稀疏乘積/哈希技巧控制記憶體）；
* **兩視角/多群構造**：原序列 vs. 隨機替換率 $p\in\{0.01,0.05,0.1\}$ 的擾動序列為不同視角/群；
* **學習**：兩視角 CCA 或 MG-TCCA 萃取共享子空間；在該子空間上做線性分類（LR/SVM）＋交叉驗證；
* **評估**：Macro-F1/ROC-AUC、對 $p$ 的穩健性曲線、在 PolySimDNA 上做**可控消融**（改 $k,d$、是否 CCA、是否多群）；
* **發布**：PolySimDNA 生成器與 CPU 可執行腳本（含隨機種子）。

---

# 參考與背景（精選）

* TCBB 的範疇重視演算法與數理方法。 ([Scimago][8])
* k-mer 在生資的地位與近期回顧。 ([PMC][2])
* UCI **Promoter** 與 **Splice** 小型經典資料集。 ([archive.ics.uci.edu][1])
* **序列錯誤對 k-mer 的影響**（替換錯誤會改變所有重疊 k-mers）。 ([BioMed Central][3])
* **pCCA/白化-CCA 理論**（共享訊號在加性獨立噪聲下可被 CCA 回收）。 ([CiteSeerX][4], [PMC][5])
* **MG-TCCA**（多群共享訊號學習）。 ([ACM Digital Library][6], [PMC][7])

---

[1]: https://archive.ics.uci.edu/ml/datasets/Molecular%2BBiology%2B%28Promoter%2BGene%2BSequences%29?utm_source=chatgpt.com "Molecular Biology (Promoter Gene Sequences)"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11152613/?utm_source=chatgpt.com "A survey of k-mer methods and applications in bioinformatics"
[3]: https://genomebiology.biomedcentral.com/articles/10.1186/gb-2010-11-11-r116?utm_source=chatgpt.com "Quake: quality-aware detection and correction of sequencing ..."
[4]: https://citeseerx.ist.psu.edu/document?doi=8c8233a4560d00111b5436fd5d0c56a90e061708&repid=rep1&type=pdf&utm_source=chatgpt.com "A Probabilistic Interpretation of Canonical Correlation ..."
[5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6327589/?utm_source=chatgpt.com "A whitening approach to probabilistic canonical correlation ..."
[6]: https://dl.acm.org/doi/10.1145/3584371.3612962?utm_source=chatgpt.com "Multi-Group Tensor Canonical Correlation Analysis"
[7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10593155/?utm_source=chatgpt.com "Multi-Group Tensor Canonical Correlation Analysis - PMC"
[8]: https://www.scimagojr.com/journalsearch.php?q=17971&tip=sid&utm_source=chatgpt.com "IEEE/ACM Transactions on Computational Biology and ..."
[9]: https://search.r-project.org/CRAN/refmans/DMRnet/html/promoter.html?utm_source=chatgpt.com "promoter dataset"
[10]: https://archive.ics.uci.edu/dataset/69/molecular%2Bbiology%2Bsplice%2Bjunction%2Bgene%2Bsequences?utm_source=chatgpt.com "Molecular Biology (Splice-junction Gene Sequences)"
[11]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10178302/?utm_source=chatgpt.com "Topological feature generation for link prediction in ..."
[12]: https://www.researchgate.net/publication/384097268_Topological-Similarity_Based_Canonical_Representations_for_Biological_Link_Prediction?utm_source=chatgpt.com "Topological-Similarity Based Canonical Representations ..."
[13]: https://arxiv.org/abs/2412.03744?utm_source=chatgpt.com "A novel approach to differential expression analysis of co-occurrence networks for small-sampled microbiome data"
[14]: https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.STACS.2024.43?utm_source=chatgpt.com "A Faster Algorithm for Constructing the Frequency Difference ..."
