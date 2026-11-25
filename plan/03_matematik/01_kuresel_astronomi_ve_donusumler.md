# Küresel astronomi ve koordinat dönüşümleri

Hedef modüller: `temel/aci.py`, `temel/vektor.py`, `koordinat/`.

Denklemler planlama düzeyinde, hatırlanan biçimleriyle yazılmıştır; her biri kodlanmadan önce anılan kaynağın ilgili bölümünden okunup `@kaynak` kaydına denklem numarasıyla işlenir.

## 1. Açı aritmetiği

- Birimler: radyan (iç gösterim), derece, saat-dakika-saniye, yay dakikası, yay saniyesi, miliyay saniyesi. `1 sa = 15°`, `1″ = π/648000 rad`.
- Sarmalama: `[0, 2π)` ve `[-π, π)` aralıklarına indirgeme; `fmod` işaret davranışı sınır testlerine girer.
- Altmışlık dizgi ayrıştırma ve biçimlendirme: `-00° 30′` örneğinde eksi işaretinin korunması bilinen bir hata kaynağıdır [Meeus1998, Böl. 1].
- Küçük açı farkları: `1 - cos θ` yerine `2 sin²(θ/2)`; kayan nokta iptalinin gerekçesi [Goldberg1991].

## 2. Küresel trigonometri

Kenarları `a, b, c`, karşı açıları `A, B, C` olan küresel üçgende [Smart1977, Böl. 1]:

- Kosinüs: `cos a = cos b cos c + sin b sin c cos A`
- Sinüs: `sin a / sin A = sin b / sin B = sin c / sin C`
- Beş parça: `sin a cos B = cos b sin c − sin b cos c cos A`

**Açısal ayrıklık.** Kosinüs formülü küçük açılarda duyarlılık kaybeder. Uygulamada Vincenty biçimi kullanılır:

```
Δσ = atan2( sqrt( (cos δ2 sin Δα)² + (cos δ1 sin δ2 − sin δ1 cos δ2 cos Δα)² ),
            sin δ1 sin δ2 + cos δ1 cos δ2 cos Δα )
```

Haversin biçimi karşıt noktalar yakınında bozulur; üç biçim de yazılır ve özellik testinde birbirine karşı sınanır. Konum açısı ve verilen ayrıklık ile konum açısından hedef nokta bulma aynı modüldedir.

## 3. Vektör ve dönme matrisi yöntemi

Trigonometrik formül zinciri yerine birim vektör ve dönme matrisi kullanılır; SOFA'nın yaklaşımı budur [SOFA2023].

- Küresel → Kartezyen: `r = (cos δ cos α, cos δ sin α, sin δ)`; ters dönüşümde `atan2`.
- Temel dönmeler `R1(θ), R2(θ), R3(θ)`: eksen takımının döndürülmesi (pasif) kuralı. İşaret kuralı tek yerde tanımlanır ve testle sabitlenir.
- Dönme matrisi özellikleri: `Rᵀ R = I`, `det R = +1`. Özellik testi her bileşik matris için bunu sınar.
- Küçük dönme vektörü ve Rodrigues formülü.

## 4. Koordinat sistemleri

Her sistem bir temel düzlem ve bir başlangıç doğrultusuyla tanımlanır.

- **Ufuk (alt-az):** `sin h = sin φ sin δ + cos φ cos δ cos H`; azimut için `atan2` biçimi. Azimutun kuzeyden doğuya ölçülmesi kuralı sabittir; Meeus güneyden ölçer [Meeus1998, Böl. 13], fark notta yazılır.
- **Ekvatoral (α, δ):** saat açısı `H = LST − α`.
- **Ekliptik (λ, β):** eğiklik `ε` çevresinde `R1(ε)` dönmesi.
- **Galaktik (l, b):** ICRS'ten sabit bir dönme matrisi. Matris kuzey galaktik kutbun konumu ve galaktik merkezin konum açısından kurulur; sabitlerin kaynağı Hipparcos kataloğudur [Perryman1997]; matrisin tam duyarlıklı değerleri ERFA `icrs2g.c` dosyasından okunmuş ve uygulanmıştır [ERFA2024].
- **Süpergalaktik:** galaktikten sabit dönme.

## 5. Gözlemciye indirgeme zinciri

Katalog konumundan (ICRS, dönem J2000 ya da Gaia için J2016) gözlenen yöne giden zincir [Kaplan2005; UrbanSeidelmann2013, Böl. 7; SOFA2023]:

1. **Uzay hareketi:** öz hareket, paralaks ve dikine hızla dönemin ilerletilmesi. Doğrusal yaklaşım ve tam vektör biçimi ayrı fonksiyonlardır.
2. **Paralaks:** gözlemcinin barisentrik konumuna göre kayma; büyüklüğü `π″ = 1/d_pc`.
3. **Işığın bükülmesi:** Güneş'in çekim alanında sapma; Güneş kenarında yaklaşık 1,75″, 90° uzanımda yaklaşık 4 mas.
4. **Yıllık aberasyon:** `v/c` mertebesinde, en çok yaklaşık 20,5″; görelilikli tam biçim.
5. **Çerçeve yanlılığı, presesyon, nutasyon:** GCRS'ten tarihin gerçek ekvator ve ekinoksuna. Modeller: IAU 2006 presesyon [Capitaine2003] ve IAU 2000A/2000B nutasyon [IERS2010, Böl. 5]. Ekinoks tabanlı ve CIO tabanlı iki yol aynı sonucu vermelidir.
6. **Dünya dönmesi:** Dünya Dönme Açısı (ERA) ya da Greenwich görünür yıldız zamanı.
7. **Kutup hareketi:** `xp, yp` (IERS Bülten A/B).
8. **Günlük aberasyon ve günlük paralaks:** gözlemcinin yermerkezli konum ve hızı.
9. **Atmosferik kırılma:** gözlenen yüksekliğe.

Her adım ayrı, saf bir fonksiyondur; zincir bunları birleştiren ince bir katmandır. Doğruluk hedefleri üç düzeyde tanımlanır:

- Düzey 1 (yaklaşık 1′): düşük duyarlıklı formüller [Meeus1998]; teleskop yönlendirme ve planlama için yeter.
- Düzey 2 (yaklaşık 1 mas): IAU 2000B nutasyon, tam aberasyon.
- Düzey 3 (mikroyay saniyesi): IAU 2006/2000A, ERFA ile `1e-10` rad düzeyinde uyum.

Sıra Düzey 1'den başlar; her düzey bir öncekini test kâhini olarak kullanır.

## 6. Presesyon ve nutasyon

- **Presesyon:** `ζ, z, θ` açılarıyla klasik biçim (IAU 1976) ve Fukushima-Williams açılarıyla IAU 2006 biçimi [Capitaine2003]. Zaman argümanı TT'de Jülyen yüzyılı: `T = (JD_TT − 2451545.0) / 36525`.
- **Ortalama eğiklik:** `ε0` IAU 2006 polinomu; başlangıç değeri 84381,406″ [IERS2010, Böl. 5].
- **Nutasyon:** Ay-Güneş ve gezegen terimleri, Delaunay argümanlarının doğrusal birleşimleri üzerinden trigonometrik seri. 2000A modeli 1365 terim, 2000B kısaltması 77 terim içerir. Katsayı tablosu elle yazılmaz; IERS'in yayımladığı elektronik tablodan üretilir ve dosyanın özeti kaydedilir.
- **Ekinoksların denklemi:** `GAST − GMST = Δψ cos ε` ve tamamlayıcı terimler.

## 7. Atmosferik kırılma

- Düzlem-paralel yaklaşım: `R ≈ (n − 1) tan z`.
- Bennett formülü: görünür yükseklik `h` (derece) için `R′ = cot( h + 7.31/(h + 4.4) )` [Bennett1982]. Basınç ve sıcaklık düzeltmesi `(P/1010 hPa) · (283 K/T)` çarpanıdır.
- Ufka yakın bölgede formüllerin doğruluğu düşer; geçerlilik aralığı belge dizgisine yazılır [Young1994].
- Kırılma dalga boyuna bağlıdır; atmosferik dağılma tayf ve astrometri için ayrı fonksiyondur.

## 8. Görüntü düzlemine izdüşüm

FITS WCS standardının izdüşümleri [GreisenCalabretta2002; CalabrettaGreisen2002]:

- Gnomonik (TAN): `ξ = cos δ sin Δα / cos c`, `η = (cos δ0 sin δ − sin δ0 cos δ cos Δα) / cos c`, `cos c = sin δ0 sin δ + cos δ0 cos δ cos Δα`. Çoğu CCD görüntüsünün izdüşümü budur.
- SIN, STG, ARC, ZEA (zenital aile); CAR, MER, CEA (silindirik); AIT, MOL (tüm gökyüzü).
- Piksel → ara koordinat doğrusal dönüşümü: `CRPIX`, `CD` ya da `PC` + `CDELT` matrisi.
- Bozulma polinomu: SIP kuralı [Shupe2005].

Gidiş-dönüş testi her izdüşüm için piksel → gök → piksel hatasının `1e-9` pikselin altında kaldığını sınar.

## 9. Küre üzerinde ayrıklaştırma

- HEALPix: eşit alanlı, hiyerarşik pikselleme; halka ve iç içe numaralama; `nside`, piksel alanı `4π/(12 nside²)` [Gorski2005].
- Çapraz eşleme: küresel k-d ağacı ya da HEALPix kovalamasıyla yarıçap içi komşu arama.

## Uygulama sırası

`aci` → `vektor` → küresel dönüşümler → Düzey 1 presesyon/nutasyon → ufuk dönüşümü ve kırılma → TAN izdüşümü → Düzey 2 → Düzey 3 → diğer izdüşümler → HEALPix.
