# İstatistik ve çıkarım

Hedef modül: `istatistik/`.

Gözlemden fiziksel parametreye giden her adım bir çıkarım problemidir: Hubble sabiti, bir geçişin derinliği ya da bir kümenin uzaklığı, verinin bir model altındaki olasılığından elde edilir. Denklemler kodlanmadan önce anılan kaynaktan okunur.

## 1. Olasılık temelleri

- Dağılımlar: Gauss, Poisson, binom, `χ²`, Student-t, log-normal, üstel, Cauchy. Her biri için yoğunluk, birikimli dağılım, momentler ve örnekleme [Ivezic2020, Böl. 3].
- Poisson istatistiği: foton sayımında `σ = √N`; düşük sayımlarda Gauss yaklaşımının bozulması ve Cash istatistiği [Cash1979].
- Merkezi limit teoremi ve geçerli olmadığı durumlar (ağır kuyruk).
- Bayes teoremi: `p(θ|D) ∝ p(D|θ) p(θ)` [SiviaSkilling2006, Böl. 1].

## 2. Belirsizlik yayılımı

- Doğrusal yayılım: `Σ_f = J Σ_x Jᵀ`. Jacobian analitik, sonlu fark ya da otomatik türevle.
- Doğrusal olmayan durumda yanlılık: kadir ↔ akı, paralaks → uzaklık. `1/ϖ` yansız bir uzaklık kestirimi değildir; göreli paralaks hatası büyüdükçe önsel dağılımla tam çıkarım gerekir [BailerJones2015; Luri2018].
- Monte Carlo yayılımı ve doğrusal yayılımla tutarlılık testi.
- Sistematik ve istatistik belirsizliğin ayrı tutulması; kovaryans matrisine sistematik terim eklenmesi [Conley2011; Brout2022].

## 3. Sağlam istatistik

- Konum: medyan, kırpılmış ortalama, çift ağırlıklı (biweight) konum. Ölçek: MAD (`σ ≈ 1,4826 · MAD`), çift ağırlıklı ölçek [Beers1990].
- Sigma kırpma: yinelemeli; merkez ve ölçek kestiricisi seçilebilir. Görüntü birleştirme ve arka plan kestiriminin temel aracıdır.
- Aykırı değerlerin atılması yerine modellenmesi: karışım modeli [HoggBovyLang2010, Böl. 3].

## 4. Model uydurma

- En büyük olabilirlik ve Gauss gürültüsünde `χ²` ile eşdeğerliği.
- Doğrusal en küçük kareler: ağırlıklı, kovaryanslı; kapalı çözüm ve parametre kovaryansı [HoggBovyLang2010, Böl. 1].
- **İki eksende hata ve iç saçılma:** astronomide olağan durumdur (Leavitt yasası, Tully-Fisher, `M-σ`). Yöntemler: Akritas-Bershady BCES [AkritasBershady1996], Bayesçi hiyerarşik model [Kelly2007], doğruya dik uzaklık olabilirliği [HoggBovyLang2010, Böl. 7]. Sıradan en küçük kareler burada eğimi yanlı verir.
- Uyum iyiliği: `χ²`, indirgenmiş `χ²` ve sınırları (doğrusal olmayan modelde serbestlik derecesi belirsizdir) [Andrae2010]; artık çözümlemesi.
- Model karşılaştırma: olabilirlik oranı, AIC, BIC, Bayes kanıtı. Hangisinin hangi soruya yanıt verdiği belgelenir [Ivezic2020, Böl. 5].

## 5. Bayesçi örnekleme

- **Metropolis-Hastings:** öneri dağılımı, kabul oranı, ayrıntılı denge [Metropolis1953; Hastings1970].
- **Afin değişmez topluluk örnekleyici:** gerdirme hamlesi [GoodmanWeare2010]; `emcee`'nin algoritması [ForemanMackey2013]. Yeniden yazılır ve `emcee` ile aynı hedef dağılımda momentleri karşılaştırılır.
- **Hamilton Monte Carlo ve NUTS:** gradyan gerektirir; yüksek boyutta verimlidir [HoffmanGelman2014]. İlk aşamada yazılmaz; NumPyro ya da PyMC kâhin ve üretim aracı olarak kullanılır.
- **İç içe örnekleme:** kanıt hesabı ve çok tepeli sonsal dağılımlar [Skilling2006; Speagle2020].
- **Yakınsama tanıları:** sıra-normalleştirilmiş bölünmüş `R̂`, etkin örnek sayısı, öz ilinti süresi [GelmanRubin1992; Vehtari2021; ForemanMackey2013]. Zincir izleri ve köşe grafikleri her analizde üretilir.
- Uygulama kuralları [HoggForemanMackey2018]: başlatma, ısınma evresinin atılması, önsel dağılımın açıkça yazılması, parametrelemenin etkisi.

## 6. Önsel dağılımlar ve hiyerarşik modeller

- Bilgi vermeyen önsel yoktur: ölçek parametresinde düz ve log-düz önsel farklı sonuç verir [SiviaSkilling2006, Böl. 5].
- Hiyerarşik model: popülasyon dağılımı ile tek tek ölçümlerin birlikte modellenmesi (iç saçılma, seçim etkisi) [Gelman2013, Böl. 5].
- Sonsal öngörü denetimi: modelden üretilen yapay verinin gözlenenle karşılaştırılması [Gelman2013, Böl. 6].

## 7. Seçim etkileri ve yanlılıklar

- **Malmquist yanlılığı:** akı sınırlı örneklemde uzaktaki kaynakların ortalamadan parlak olması; Hubble diyagramında uzaklığı küçültür.
- **Eddington yanlılığı:** dik bir sayım dağılımında ölçüm hatasının sınırın üstüne taşıdığı kaynaklar.
- **Lutz-Kelker yanlılığı:** paralaks eşiğiyle seçilmiş örneklemde [Luri2018].
- Düzeltme yolları: ileri modelleme (seçim fonksiyonu olabilirliğe girer), benzetimle yanlılık düzeltmesi [KesslerScolnic2017], kesilmiş ve sansürlü veri olabilirliği [Ivezic2020, Böl. 4].

Seçim etkisi Hubble diyagramı analizinin baskın sistematiklerindendir; `kozmoloji` modülü bunu başından tasarıma alır.

## 8. Parametrik olmayan ve yeniden örnekleme yöntemleri

- Bootstrap ve jackknife [Efron1979]; ilintili veride blok bootstrap.
- Çekirdek yoğunluk kestirimi, histogram kutu genişliği kuralları, Bayesçi bloklar [Ivezic2020, Böl. 5-6].
- Kolmogorov-Smirnov ve Anderson-Darling testleri ve yanlış kullanımları (parametreleri veriden kestirilmiş dağılıma karşı).

## 9. Gauss süreçleri

İlintili gürültünün (yıldız değişkenliği, aygıt sistematikleri) modeli [RasmussenWilliams2006, Böl. 2]. Genel çözüm `O(N³)` ölçeklenir; tek boyutlu zaman serisinde `celerite` çekirdek ailesi `O(N)` çözüm verir [ForemanMackey2017]. İlk aşamada küçük `N` için Cholesky tabanlı genel çözüm yazılır.

## 10. Makine öğrenmesi

Sınıflandırma (yıldız/gökada ayrımı, değişen yıldız türleri), regresyon (fotometrik kırmızıya kayma), kümeleme (açık küme üyeliği: Gaia öz hareket ve paralaks uzayında), boyut indirgeme [Ivezic2020, Böl. 6-9]. Bu yöntemler yeniden yazılmaz; `scikit-learn` kullanılır. Projenin katkısı öznitelik çıkarımı, eğitim/sınama ayrımının doğru kurulması ve belirsizliğin taşınmasıdır.

Benzetime dayalı çıkarım (olabilirliğin yazılamadığı durumlar) ileri aşama konusudur [Cranmer2020].

## Uygulama sırası

Dağılımlar ve sağlam istatistik → doğrusal uyum (kovaryanslı) → belirsizlik yayılımı → Metropolis-Hastings → topluluk örnekleyici ve tanılar → iki eksende hatalı uyum → seçim etkili olabilirlik → iç içe örnekleme → Gauss süreci.
