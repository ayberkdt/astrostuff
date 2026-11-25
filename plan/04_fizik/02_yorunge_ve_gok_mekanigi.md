# Yörünge ve gök mekaniği

Hedef modül: `efemeris/`.

Denklemler planlama düzeyindedir; kodlanmadan önce anılan kaynaktan okunur.

## 1. İki cisim problemi

- Göreli hareket denklemi: `r̈ = −μ r / r³`, `μ = G(m1 + m2)`.
- Korunan nicelikler: özgül enerji `ε = v²/2 − μ/r`, açısal momentum `h = r × v`, dışmerkezlik (Laplace-Runge-Lenz) vektörü [MurrayDermott1999, Böl. 2].
- Yörünge denklemi: `r = a(1 − e²) / (1 + e cos f)`; vis-viva: `v² = μ(2/r − 1/a)`.
- Kepler'in üçüncü yasası: `P² = 4π² a³ / μ`.
- `GM` çarpımı `G` ve `M` ayrı ayrı bilindiğinden çok daha duyarlı ölçülür; sabitler dosyasında nominal `GM` değerleri tutulur [Prsa2016].

## 2. Anomaliler ve Kepler denklemi

- Ortalama anomali `M = n(t − t0)`, dış merkezli anomali `E`, gerçek anomali `f`.
- Kepler denklemi `M = E − e sin E`; çözücüler [../03_matematik/03_sayisal_yontemler.md](../03_matematik/03_sayisal_yontemler.md) dosyasında [Markley1995].
- `E → f`: `tan(f/2) = sqrt((1+e)/(1−e)) tan(E/2)`; `atan2` biçimi tercih edilir.
- Parabolik (Barker denklemi) ve hiperbolik yörüngeler; evrensel değişken biçimi [Vallado2022, Böl. 2].

## 3. Yörünge elemanları ve durum vektörü

- Klasik elemanlar: `a, e, i, Ω, ω, M0` ve dönem.
- Elemanlar ↔ durum vektörü (`r, v`) dönüşümü; perifokal çerçeveden `R3(−Ω) R1(−i) R3(−ω)` dönmesi [Vallado2022, Böl. 2].
- Tekillikler: `e = 0` ve `i = 0` durumunda bazı açılar tanımsızdır; ekinoktal elemanlar bu durumları kaldırır.
- Ortalama ve anlık (osküle) elemanların ayrımı.

## 4. Güneş, Ay ve gezegen konumları

Üç doğruluk düzeyi:

- **Düzey 1:** kısaltılmış seriler. Güneş için yaklaşık 0,01°, Ay için yaklaşık 10″, gezegenler için ortalama elemanlar [Meeus1998, Böl. 25, 31-33, 47]. Planlama ve doğuş-batış için yeter.
- **Düzey 2:** tam analitik kuramların (VSOP87, ELP) terim tabloları [Bretagnon1988]; ELP kuramının künyesi eklenecek.
- **Düzey 3:** JPL DE440 sayısal efemerisinin okunması: SPK dosya biçimi, Chebyshev katsayılarının değerlendirilmesi [Park2021]. DE440 1550-2650 yıllarını kapsar; DE441 daha uzun aralığı daha düşük doğrulukla verir. Bu aralık bilgisi JPL'in `de440_de441` belge sayfasından araştırma sırasında okunmuştur.

Kâhinler: Skyfield ve `jplephem` (aynı DE dosyasıyla bit düzeyine yakın uyum beklenir), JPL Horizons (bağımsız hizmet).

## 5. Küçük cisimler ve yapay uydular

- Asteroit ve kuyruklu yıldız efemerisi: Minor Planet Center elemanlarından iki cisim ilerletme ve gezegen tedirginlikleriyle sayısal integrasyon.
- Kadir modeli: `H, G` sistemi.
- Yapay uydular: TLE biçimi ve SGP4 ilerletici [Vallado2022, Böl. 9]. SGP4 yeniden yazılmaz; `sgp4` paketi kullanılır, çünkü TLE'ler yalnızca o modelin tanımıyla anlamlıdır. Görüntülerdeki uydu izlerinin öngörülmesi için gerekir.

## 6. Yörünge belirleme

- Üç gözlemden başlangıç yörüngesi: Gauss ve Laplace yöntemleri [Vallado2022, Böl. 7].
- Diferansiyel düzeltme: ağırlıklı en küçük kareler, durum geçiş matrisi.
- Görsel ve tayfsal çift yıldız yörüngeleri: Thiele-Innes sabitleri, dikine hız eğrisi `v_r = K[cos(ω + f) + e cos ω] + γ`, kütle fonksiyonu.

## 7. N cisim ve tedirginlik

- Doğrudan toplam `O(N²)`; yumuşatma uzunluğu.
- İntegratörler: leapfrog, Wisdom-Holman [WisdomHolman1991; ReinTamayo2015], yüksek dereceli uyarlamalı [ReinSpiegel2015]. Kâhin: REBOUND [ReinLiu2012].
- Kısıtlı üç cisim problemi: Jacobi sabiti, Lagrange noktaları, Hill küresi, Roche limiti [MurrayDermott1999, Böl. 3].
- Seküler tedirginlikler, rezonanslar, gelgit evrimi (ileri aşama) [MurrayDermott1999, Böl. 6-8].
- Görelilikli düzeltmeler: günberi ilerlemesi; birinci post-Newton terimi.

## 8. Ötegezegen gözlenebilirleri

- Geçiş geometrisi: derinlik `(Rp/R*)²`, süre, çarpma parametresi; ışık eğrisinden yıldız yoğunluğu [SeagerMallenOrnelas2003; Winn2010].
- Geçiş ışık eğrisi modeli: kenar kararmalı disk üzerinde analitik çözüm [MandelAgol2002]. Kâhin: `batman` [Kreidberg2015].
- Dikine hız yarı genliği `K` ve en küçük kütle `m sin i`.
- Geçiş zamanlama değişimleri (ileri aşama).

## Test durumları

- Kapalı çözüm: dairesel ve eliptik yörüngede bir dönem sonunda başlangıç durumuna dönüş; enerji ve açısal momentum korunumu.
- Kaynak örnekleri [Meeus1998; Vallado2022] sayısal olarak yeniden üretilir.
- Gezegen konumları Skyfield + DE440 ile, Düzey 1'de kaynağın bildirdiği doğruluk içinde, Düzey 3'te mikroyay saniyesi düzeyinde karşılaştırılır.
- N cisim: REBOUND ile aynı başlangıç koşullarında yörünge farkı; figür-sekiz üç cisim çözümünün kararlılığı.
