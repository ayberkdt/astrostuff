# Sayısal yöntemler

Hedef modül: `temel/sayisal/`.

SciPy çalışma zamanı bağımlılığıdır, dolayısıyla genel amaçlı integral, kök bulma ve en iyileme oradan kullanılabilir. Aşağıdaki yöntemler yine de yazılır: astronomiye özgü olanlar (Kepler çözücü, simplektik integratör) SciPy'de yoktur, temel olanlar ise öğrenme amacıyla yazılıp SciPy'ye karşı sınanır. Hangi fonksiyonun üretimde kullanılacağı her modülde açıkça seçilir.

## 1. Kayan nokta aritmetiği

- IEEE 754 çift duyarlık: yaklaşık 15,95 ondalık basamak, `ε ≈ 2,22e-16` [Goldberg1991].
- İptal hatası: yakın sayıların farkı. Örnekler: küçük açıda `1 − cos θ`, küçük `z`'de uzaklık farkları, JD'nin tek `float64` ile tutulması.
- Dengeli toplama: Kahan ve Neumaier toplaması; uzun serilerin (nutasyon) toplanmasında [Higham2002, Böl. 4].
- `log1p`, `expm1`, `hypot`, `atan2` kullanımı kural olarak yazılır.
- Koşul sayısı ve geriye dönük kararlılık: bir sonucun kaç basamağına güvenilebileceğinin ölçüsü [Higham2002, Böl. 1].

## 2. Kök bulma

- İkiye bölme, Newton-Raphson, sekant, Brent yöntemi [Press2007, Böl. 9].
- **Kepler denklemi** `M = E − e sin E`: Newton ve Halley yinelemesi, başlangıç tahmini seçimi, `e → 1` yakınında yakınsama. Yinelemesiz çözüm: Markley'in beşinci dereceden düzeltmeli yöntemi [Markley1995]. Hiperbolik (`M = e sinh H − H`) ve evrensel değişkenli biçim [Vallado2022, Böl. 2].
- Test: `[0, 2π) × [0, 1)` ızgarasında `|E − e sin E − M| < 1e-14`.

## 3. Sayısal integral

- Yamuk, Simpson, Romberg; Gauss-Legendre ve uyarlamalı Gauss-Kronrod [Press2007, Böl. 4].
- Kozmolojik uzaklık integrali `∫ dz / E(z)`: düzgün integrand, sabit dereceli Gauss-Legendre yeter; vektörleştirilmiş birikimli integral (çok sayıda `z` için tek geçiş) ayrı fonksiyondur.
- Sentetik fotometri integralleri: tablolanmış tayf ve filtre eğrisi üzerinde yamuk kuralı; ızgara yeniden örnekleme hatası ölçülür.
- Tekil ya da salınımlı integrandlar için değişken dönüşümü.

## 4. Adi diferansiyel denklemler

- Açık Runge-Kutta: RK4; gömülü çiftle adım denetimi, Dormand-Prince 5(4) [DormandPrince1980; Hairer1993, Böl. II].
- **Simplektik yöntemler:** leapfrog (kick-drift-kick), Yoshida'nın dördüncü dereceden bileşimi [Yoshida1990]. Hamilton sistemlerinde enerji hatası sınırlı kalır; RK4'te birikir. Karşılaştırma testi Kepler yörüngesinde 10⁴ dönem boyunca enerji hatasını ölçer.
- **Wisdom-Holman haritası:** Kepler hareketi ile gezegenler arası etkileşimin ayrıştırılması [WisdomHolman1991; ReinTamayo2015].
- **Yüksek dereceli uyarlamalı integratör:** Gauss-Radau aralıklı 15. derece yöntem, yakın geçişler için [ReinSpiegel2015].
- Katı sistemler (yıldız yapısı, tepkime ağları): kapalı yöntemler, SciPy `solve_ivp` (`Radau`, `BDF`) üzerinden.
- İki noktalı sınır değer problemi: atış yöntemi ve gevşetme; Lane-Emden ve yıldız yapısı denklemleri için.

## 5. İnterpolasyon ve yaklaşım

- Doğrusal, kübik spline, Lagrange, Chebyshev [Press2007, Böl. 3 ve 5].
- **Chebyshev efemerisi:** JPL DE dosyaları konumu zaman aralıkları üzerinde Chebyshev katsayılarıyla saklar; değerlendirme Clenshaw özyinelemesiyle yapılır, hız türev serisinden gelir [Park2021].
- Görüntü yeniden örnekleme: en yakın komşu, çift doğrusal, çift kübik, Lanczos çekirdeği; akı korunumu ve gürültü ilintisi üzerindeki etkisi.
- Tayfın yeniden örneklenmesi: akı koruyan yeniden kutulama.

## 6. Doğrusal cebir

- En küçük kareler: normal denklemler yerine QR ya da SVD; koşul sayısı raporlanır [Press2007, Böl. 15].
- Kovaryans matrisiyle `χ² = rᵀ C⁻¹ r`: ters alınmaz; Cholesky ayrışımı ve üçgen çözümle hesaplanır. Pantheon+ kovaryansı 1701 × 1701 boyutundadır [Brout2022]; bu yol hem daha hızlı hem daha kararlıdır.
- Temel bileşen analizi: SVD üzerinden.

## 7. Fourier yöntemleri

- Ayrık Fourier dönüşümü, örnekleme teoremi, Nyquist frekansı, örtüşme, sızıntı ve pencere fonksiyonları [Press2007, Böl. 12-13].
- Evrişim ve çapraz ilinti FFT ile: PSF evrişimi, görüntü hizalama (faz ilintisi), tayfsal kırmızıya kayma ölçümü [TonryDavis1979].
- Eşit aralıklı olmayan örnekleme: Lomb-Scargle [VanderPlas2018]; ayrıntı [../05_veri_bilimi/04_zaman_serisi.md](../05_veri_bilimi/04_zaman_serisi.md) dosyasında.

## 8. En iyileme

- Doğrusal olmayan en küçük kareler: Gauss-Newton ve Levenberg-Marquardt; Jacobian'ın analitik ya da sonlu farkla hesabı [Press2007, Böl. 15].
- Türevsiz yöntem: Nelder-Mead. Yarı-Newton: BFGS.
- Parametre kovaryansı: en iyi noktada `(JᵀWJ)⁻¹`; yalnızca olabilirlik yaklaşık Gauss ise geçerlidir.

## 9. Rastgele sayılar ve Monte Carlo

- `numpy.random.Generator` (PCG64); üreteç argüman olarak geçirilir.
- Ters dönüşüm, ret örneklemesi, önem örneklemesi [Press2007, Böl. 7].
- Belirsizlik yayılımı için Monte Carlo ve doğrusal yayılımla karşılaştırma.

## 10. Özel fonksiyonlar

Eliptik integraller (geçiş modeli [MandelAgol2002]), gama ve tamamlanmamış gama (`χ²` olasılığı), hata fonksiyonu, Bessel fonksiyonları (Airy deseni), Legendre polinomları ve küresel harmonikler (çekim alanı, CMB). Tümü `scipy.special` üzerinden kullanılır; yeniden yazılmaz.

## Doğrulama

Her yöntem kapalı çözümü bilinen problemde yakınsama derecesiyle sınanır: adım yarıya indiğinde hata `2^p` kat düşmelidir. Yakınsama derecesi testi, uygulamanın yalnızca çalıştığını değil kuramın öngördüğü hızda yakınsadığını gösterir.
