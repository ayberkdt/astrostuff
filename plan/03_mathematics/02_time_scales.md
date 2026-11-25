# Zaman ölçekleri

Hedef modül: `zaman/`.

Astronomide tek bir "zaman" yoktur; her ölçek başka bir fiziksel tanıma dayanır ve yanlış ölçek kullanmak bir dakikayı aşan hata üretir [Eastman2010]. Denklemler kodlanmadan önce anılan kaynaktan okunur.

## 1. Takvim ve Jülyen tarihi

- Jülyen Tarihi (JD): MÖ 4713 1 Ocak öğlesinden bu yana geçen gün sayısı. Değiştirilmiş JD: `MJD = JD − 2400000.5`.
- Gregoryen ↔ JD dönüşümü: tamsayı aritmetiğiyle algoritma [Meeus1998, Böl. 7]. Jülyen-Gregoryen geçişi (4 Ekim 1582'yi 15 Ekim izler) ve eksi yıllar test durumudur.
- Dönemler: `J2000.0 = JD 2451545.0 (TT)`; Jülyen yılı 365,25 gün; Bessel dönemi eski kataloglar için.
- **İki parçalı gösterim.** JD iki `float64` olarak tutulur (`jd1 + jd2`). Tek `float64` J2000 yakınında yaklaşık 40 µs çözünürlük verir; iki parça bunu nanosaniyenin altına indirir [SOFA2023].

## 2. Ölçekler

- **TAI:** atomik zaman; SI saniyesinin sürekli sayımı.
- **UTC:** TAI'den tam saniye farklı sivil zaman; artık saniyelerle UT1'e 0,9 s içinde tutulur. `TAI − UTC` artık saniye tablosundan okunur; tablo IERS Bülten C ile güncellenir ve koda gömülü kopyasının son geçerlilik tarihi vardır. Artık saniye anında (`23:59:60`) davranış ayrıca test edilir.
- **UT1:** Dünya'nın dönme açısına bağlı zaman. `UT1 − UTC` (DUT1) IERS Bülten A/B'den okunur; öngörülemez, ölçülür.
- **TT:** yermerkezli efemeris zamanı. `TT = TAI + 32,184 s`.
- **TDB:** barisentrik dinamik zaman; efemerislerin argümanı. `TDB − TT` genliği yaklaşık 1,7 ms olan dönemsel bir farktır; düşük duyarlıklı tek terimli biçim ve tam seri ayrı fonksiyonlardır [Kaplan2005; SOFA2023].
- **TCG, TCB:** görelilikli koordinat zamanları; TT ve TDB'ye doğrusal bağlıdır [IERS2010, Böl. 10].
- **GPS zamanı:** `TAI − 19 s`.
- **ΔT = TT − UT1:** tarihsel dönemler için polinom yaklaşımları; gelecek için tahmin.

Dönüşüm çizgesi: `UTC ↔ TAI ↔ TT ↔ TDB`, `UTC ↔ UT1`, `TT ↔ TCG`, `TDB ↔ TCB`. Her kenar bir fonksiyondur; iki ölçek arasındaki dönüşüm çizgedeki en kısa yoldan kurulur.

## 3. Yıldız zamanı ve Dünya dönme açısı

- Dünya Dönme Açısı: `ERA = 2π (0,7790572732640 + 1,00273781191135448 · Tu)`, `Tu = JD_UT1 − 2451545,0` [IERS2010, Böl. 5]. Katsayılar kaynaktan teyit edilecek.
- Greenwich ortalama yıldız zamanı (GMST): ERA ile presesyona bağlı bir polinomun toplamı (IAU 2006); eski IAU 1982 biçimi düşük duyarlık düzeyi için.
- Greenwich görünür yıldız zamanı: `GAST = GMST + ekinoksların denklemi`.
- Yerel yıldız zamanı: `LST = GAST + boylam` (doğu artı).

## 4. Barisentrik düzeltme

Zamanlama (değişen yıldız, geçiş, pulsar) ve dikine hız ölçümleri Güneş sistemi kütle merkezine indirgenir.

- **Rømer gecikmesi:** `Δt = (r · n̂) / c`; `r` gözlemcinin barisentrik konumu, `n̂` kaynağın yönü. Genliği 8,3 dakikaya kadar çıkar.
- **Shapiro gecikmesi** ve **Einstein gecikmesi** (TT → TDB).
- `BJD_TDB` standart çıktıdır; `HJD` ve `BJD_UTC` ile farkları belgelenir [Eastman2010].
- Barisentrik hız düzeltmesi: 1 m/s düzeyi için basit izdüşüm, 1 cm/s için tam görelilikli biçim [WrightEastman2014].

Bu fonksiyonlar gözlemcinin barisentrik konumuna ihtiyaç duyar; `efemeris` modülüne bağımlıdır.

## 5. Olay zamanları

- Doğuş, batış, meridyen geçişi: standart yükseklik (yıldız için −34′, Güneş kenarı için −50′), yinelemeli çözüm [Meeus1998, Böl. 15].
- Alacakaranlık sınırları: sivil −6°, denizcilik −12°, astronomik −18°.
- Ay evresi ve aydınlanma oranı.

## Test durumları

- Kaynakların çalışılmış örnekleri [Meeus1998, Böl. 7 ve 12; SOFA2023 zaman ölçekleri yemek kitabı].
- ERFA ve `astropy.time` ile her dönüşümde `1e-9` s düzeyinde uyum (UT1 ve artık saniye tabloları aynı olmak koşuluyla).
- Özellik: her dönüşümün tersi gidiş-dönüşte birim işlemdir; ölçek farkları tekdüzedir.
